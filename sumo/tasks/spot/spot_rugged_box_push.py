# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np
from judo.utils.fields import np_1d_field
from mujoco import MjData, MjModel

from sumo import MODEL_PATH
from sumo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from sumo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_rugged_box.xml")

X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])

# annulus object position sampling
RADIUS_MIN = 1.5
RADIUS_MAX = 3.0
USE_LEGS = False
USE_GRIPPER = True
USE_TORSO = True

HARDWARE_FENCE_X = (-2.0, 3.0)
HARDWARE_FENCE_Y = (-3.0, 2.5)

DEFAULT_SPOT_POS = np.array([0.0, 0.0])
DEFAULT_OBJECT_POS = np.array([3.0, 0.0])


@dataclass
class SpotRuggedBoxPushConfig(SpotBaseConfig):
    """Config for the spot rugged box push task."""

    goal_position: np.ndarray = np_1d_field(
        np.array([-0.0, 0.0, 0.0], dtype=np.float64),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, -1.0],
        maxs=[5.0, 5.0, 2.0],
        steps=[0.1, 0.1, 0.1],
        vis_name="rugged_box_goal_position",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[-1.0, 1.3, 0.0],
    )
    w_fence: float = 1000.0
    w_goal: float = 500.0
    w_orientation: float = 50.0
    w_torso_proximity: float = 30.0
    w_gripper_proximity: float = 0.1
    w_controls: float = 2.0
    orientation_threshold: float = 0.5
    torso_distance_from_box: float = 0.6  # Distance from box for desired torso position
    torso_desired_height: float = 0.4  # Desired height for torso position
    torso_height_upper_bound: float = 0.4  # Upper bound for torso height control
    position_tolerance: float = 0.2


class SpotRuggedBoxPush(SpotBase):
    """Task getting Spot to push a rugged box to a desired goal location."""

    name = "spot_rugged_box_push"
    config_t = SpotRuggedBoxPushConfig
    config: SpotRuggedBoxPushConfig

    def __init__(self, config: SpotRuggedBoxPushConfig | None = None) -> None:
        super().__init__(model_path=XML_PATH, use_arm=True, use_gripper=True, use_torso=True, config=config)

        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.object_pose_idx = self.get_joint_position_start_index("rugged_box_joint")
        self.object_vel_idx = self.get_joint_velocity_start_index("rugged_box_joint")

        # Object orientation sensors
        self.object_x_axis_idx = self.get_sensor_start_index("object_x_axis")
        self.object_y_axis_idx = self.get_sensor_start_index("object_y_axis")
        self.object_z_axis_idx = self.get_sensor_start_index("object_z_axis")

        # End effector to object sensor
        self.end_effector_to_object_idx = self.get_sensor_start_index("sensor_arm_link_fngr")

    @property
    def actuator_ctrlrange(self) -> np.ndarray:
        """Control bounds for the task with custom torso height upper bound.

        Override to set a lower upper bound for torso height control.
        """
        # Get the base control range from parent class
        base_ctrlrange = super().actuator_ctrlrange

        # Modify the torso height upper bound (last element in the upper bound)
        # The torso height is the last element if use_torso is True
        if self.use_torso:
            # Get custom height from config if available, otherwise use default
            custom_torso_height_upper = 0.4
            base_ctrlrange[-1, 1] = custom_torso_height_upper  # Modify upper bound of last element

        return base_ctrlrange

    def set_command_values(self) -> None:
        """Update default_command with custom torso target height."""
        super().set_command_values()
        # Set custom torso target height (default is STANDING_HEIGHT, range is 0.3 to 1.0)
        # Torso command format: [roll, pitch, height]
        if self.use_torso:
            self.default_command[-3:] = [0.0, 0.0, 0.4]  # Set height to 0.4

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the Spot rugged box moving task."""
        batch_size = states.shape[0]

        qpos = states[..., : self.model.nq]

        body_height = qpos[..., self.body_pose_idx + 2]
        body_pos = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]
        object_pos = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]

        # Sensors sampled at the same horizon as states
        object_x_axis = sensors[..., self.object_x_axis_idx : self.object_x_axis_idx + 3]
        object_y_axis = sensors[..., self.object_y_axis_idx : self.object_y_axis_idx + 3]
        object_z_axis = sensors[..., self.object_z_axis_idx : self.object_z_axis_idx + 3]
        end_effector_to_object = sensors[..., self.end_effector_to_object_idx : self.end_effector_to_object_idx + 3]

        # Fence reward
        fence_violated_x = (body_pos[..., 0] < HARDWARE_FENCE_X[0]) | (body_pos[..., 0] > HARDWARE_FENCE_X[1])
        fence_violated_y = (body_pos[..., 1] < HARDWARE_FENCE_Y[0]) | (body_pos[..., 1] > HARDWARE_FENCE_Y[1])
        spot_fence_reward = -self.config.w_fence * (fence_violated_x | fence_violated_y).any(axis=-1)

        # Check if spot has fallen
        spot_fallen_reward = -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        # Goal reward - distance to goal position
        goal_reward = -self.config.w_goal * np.linalg.norm(
            object_pos - np.array(self.config.goal_position)[None, None], axis=-1
        ).mean(-1)

        # Penalize deviation from desired orientation (all three axes)
        # Box x-axis should align with world X_AXIS
        # Box y-axis should align with world Y_AXIS
        # Box z-axis should align with world Z_AXIS
        x_alignment = np.abs(1.0 - np.sum(object_x_axis * X_AXIS, axis=-1))
        y_alignment = np.abs(1.0 - np.sum(object_y_axis * Y_AXIS, axis=-1))
        z_alignment = np.abs(1.0 - np.sum(object_z_axis * Z_AXIS, axis=-1))

        box_orientation_reward = -self.config.w_orientation * (x_alignment + y_alignment + z_alignment).mean(axis=-1)

        # Compute desired torso position: distance away from box in opposite direction of goal (2D)
        goal_position = np.array(self.config.goal_position)
        # Direction from box to goal (2D)
        box_to_goal_2d = goal_position[None, None, :2] - object_pos[..., :2]
        box_to_goal_norm = np.maximum(np.linalg.norm(box_to_goal_2d, axis=-1, keepdims=True), 1e-6)
        box_to_goal_unit = box_to_goal_2d / box_to_goal_norm

        # Desired torso position: opposite direction from box to goal
        distance_from_box = self.config.torso_distance_from_box
        torso_desired_xy = object_pos[..., :2] - distance_from_box * box_to_goal_unit
        torso_desired = np.concatenate([torso_desired_xy, object_pos[..., 2:3]], axis=-1)
        torso_desired[..., 2] = self.config.torso_desired_height

        # Compute l2 distance from torso pos. to desired torso pos.
        torso_proximity_reward = -self.config.w_torso_proximity * np.linalg.norm(
            body_pos - torso_desired, axis=-1
        ).mean(-1)

        # Compute l2 distance from end effector to object
        gripper_proximity_reward = -self.config.w_gripper_proximity * np.linalg.norm(
            end_effector_to_object,
            axis=-1,
        ).mean(-1)

        # Control penalty
        controls_reward = -self.config.w_controls * np.linalg.norm(controls[..., :3], axis=-1).mean(-1)

        assert spot_fence_reward.shape == (batch_size,)
        assert spot_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert box_orientation_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)
        assert gripper_proximity_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        return (
            # spot_fence_reward
            +spot_fallen_reward
            + goal_reward
            + box_orientation_reward
            + torso_proximity_reward
            + gripper_proximity_reward
            + controls_reward
        )

    # @property
    # def reset_pose(self) -> np.ndarray:
    #     """Reset pose of robot and object."""

    #     # Sample object position in annulus
    #     radius = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * np.random.rand()
    #     theta = 2 * np.pi * np.random.rand()
    #     object_pos = np.array([radius * np.cos(theta), radius * np.sin(theta)]) + 0.1 * np.random.randn(2)

    #     object_pose = np.array([*object_pos, 0.254, 1, 0, 0, 0])

    #     # Place robot at random x and y
    #     robot_pose_xy = np.random.uniform(-0.5, 0.5, 2)
    #     random_yaw_robot = np.random.uniform(0, 2 * np.pi)
    #     robot_pose_orientation = np.array([np.cos(random_yaw_robot / 2), 0, 0, np.sin(random_yaw_robot / 2)])
    #     robot_pose = np.array([*robot_pose_xy, STANDING_HEIGHT, *robot_pose_orientation])

    #     return np.array([*robot_pose, *LEGS_STANDING_POS, *self.reset_arm_pos, *object_pose])

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object."""
        object_pos = np.array([0.5, 2.0])
        object_pose = np.array([*object_pos, 0.0, 1, 0, 0, 0])

        # Place robot at random x and y
        robot_pose_xy = np.array([1.0, 0.0])
        robot_pose_orientation = np.array([1, 0, 0, 0])
        robot_pose = np.array([*robot_pose_xy, STANDING_HEIGHT, *robot_pose_orientation])

        return np.array([*robot_pose, *LEGS_STANDING_POS, *self.reset_arm_pos, *object_pose])

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the rugged box has been successfully pushed to the goal position."""
        object_pos = data.qpos[self.object_pose_idx : self.object_pose_idx + 3]
        goal_position = np.array(self.config.goal_position)

        # Check position tolerance
        distance_to_goal = np.linalg.norm(object_pos - goal_position)
        position_success = distance_to_goal <= self.config.position_tolerance

        return bool(position_success)
