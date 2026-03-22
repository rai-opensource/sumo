# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""SpotChairRamp task - push a chair up a ramp.

Adapted from judo/tasks/spot/spot_yellow_chair_ramp_push.py.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from judo.tasks.spot.spot_constants import Z_AXIS
from judo.utils.fields import np_1d_field
from mujoco import MjData, MjModel

from sumo import MODEL_PATH
from sumo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from sumo.tasks.spot.spot_constants import LEGS_STANDING_POS, STANDING_HEIGHT

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_yellow_chair_ramp.xml")

# Minimum distance between robot and chair centers to avoid collision at reset
MIN_ROBOT_OBJECT_DISTANCE = 1.2


@dataclass
class SpotChairRampConfig(SpotBaseConfig):
    """Configuration for the SpotChairRamp task."""

    w_goal: float = 100.0
    w_orientation: float = 100.0
    w_torso_proximity: float = 0.25
    w_gripper_proximity: float = 4.0
    orientation_threshold: float = 0.4
    w_controls: float = 2.0
    w_object_velocity: float = 128.0
    fall_penalty: float = 10000.0
    w_object_off_ramp: float = 1000.0
    w_object_centered: float = 15.0
    w_spot_off_ramp: float = 1000.0
    goal_position: np.ndarray = np_1d_field(
        np.array([2.0, 4.5, 0.5]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, None],
    )

    # Success criteria (sumo-specific)
    position_tolerance: float = 0.005


class SpotChairRamp(SpotBase[SpotChairRampConfig]):
    """Task getting Spot to push a yellow chair up a ramp."""

    name: str = "spot_chair_ramp"
    config_t: type[SpotChairRampConfig] = SpotChairRampConfig  # type: ignore[assignment]
    config: SpotChairRampConfig

    def __init__(
        self,
        config: SpotChairRampConfig | None = None,
    ) -> None:
        """Initialize the SpotChairRamp task."""
        super().__init__(model_path=XML_PATH, use_arm=True, config=config)
        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.object_pose_idx = self.get_joint_position_start_index("yellow_chair_joint")
        self.gripper_pos_idx = self.get_sensor_start_index("trace_fngr_site")
        self.object_y_axis_idx = self.get_sensor_start_index("object_y_axis")
        self.object_vel_idx = self.model.jnt_dofadr[self.model.joint("yellow_chair_joint").id]

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the chair ramp push task."""
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq :]

        body_height = qpos[..., self.body_pose_idx + 2]
        body_pos = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]
        object_pos = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        object_linear_velocity = qvel[..., self.object_vel_idx : self.object_vel_idx + 3]
        object_y_axis = sensors[..., self.object_y_axis_idx : self.object_y_axis_idx + 3]
        gripper_pos = sensors[..., self.gripper_pos_idx : self.gripper_pos_idx + 3]

        spot_fallen_reward = -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        goal_reward = -self.config.w_goal * np.linalg.norm(
            object_pos - self.config.goal_position[None, None], axis=-1
        ).mean(-1)

        object_orientation_reward = -self.config.w_orientation * (
            np.abs(np.dot(object_y_axis, Z_AXIS)) > self.config.orientation_threshold
        ).sum(axis=-1)

        torso_proximity_reward = self.config.w_torso_proximity * np.linalg.norm(body_pos - object_pos, axis=-1).mean(-1)

        gripper_proximity_reward = -self.config.w_gripper_proximity * np.linalg.norm(
            gripper_pos - object_pos, axis=-1
        ).mean(-1)

        object_linear_velocity_reward = -self.config.w_object_velocity * np.square(
            np.linalg.norm(object_linear_velocity, axis=-1).mean(-1)
        )

        # Off-ramp penalties
        object_off_ramp_x = np.logical_or(object_pos[..., 0] <= 1.0, object_pos[..., 0] >= 3.0)
        object_off_ramp_y = object_pos[..., 1] >= 5.25
        object_off_ramp = np.logical_or(object_off_ramp_x, object_off_ramp_y)
        object_off_ramp_penalty = -self.config.w_object_off_ramp * object_off_ramp.mean(-1)

        object_centered = -self.config.w_object_centered * np.abs(object_pos[..., 0] - 2.0).mean(-1)

        spot_off_ramp = body_pos[..., 1] >= 4.5
        spot_off_ramp_penalty = -self.config.w_spot_off_ramp * spot_off_ramp.mean(-1)

        controls_reward = -self.config.w_controls * np.linalg.norm(controls[..., :3], axis=-1).mean(-1)

        assert spot_fallen_reward.shape == (batch_size,)
        return (
            spot_fallen_reward
            + goal_reward
            + object_orientation_reward
            + torso_proximity_reward
            + gripper_proximity_reward
            + object_linear_velocity_reward
            + controls_reward
            + object_off_ramp_penalty
            + object_centered
            + spot_off_ramp_penalty
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose for the chair ramp push task.

        Places the chair near the ramp base and the robot behind it,
        ensuring they don't overlap.
        """
        # Place chair at the base of the ramp with small randomization
        object_xy = np.array([2.0, 0.5]) + np.random.uniform(-0.3, 0.3, 2)
        reset_object_pose = np.array([*object_xy, 0, 1, 0, 0, 0])

        # Place robot behind the chair (negative y), ensuring no overlap
        robot_xy = object_xy[:2] + np.array([0.0, -MIN_ROBOT_OBJECT_DISTANCE])
        robot_xy += np.random.uniform(-2.0, 2.0, 2)
        # Ensure minimum separation
        while np.linalg.norm(robot_xy - object_xy[:2]) < MIN_ROBOT_OBJECT_DISTANCE:
            robot_xy = object_xy[:2] + np.array([0.0, -MIN_ROBOT_OBJECT_DISTANCE])
            robot_xy += np.random.uniform(-0.2, 0.2, 2)

        return np.array(
            [*robot_xy, STANDING_HEIGHT, 1, 0, 0, 0, *LEGS_STANDING_POS, *self.reset_arm_pos, *reset_object_pose]
        )

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the yellow chair has reached the top platform area of the ramp."""
        object_pos = data.qpos[self.object_pose_idx : self.object_pose_idx + 3]

        platform_x_min, platform_x_max = 0.66, 3.09
        platform_y_min, platform_y_max = 4.192, 5.412
        platform_z_min = 0.20

        x_in_bounds = platform_x_min <= object_pos[0] <= platform_x_max
        y_in_bounds = platform_y_min <= object_pos[1] <= platform_y_max
        z_in_bounds = object_pos[2] >= platform_z_min

        return bool(x_in_bounds and y_in_bounds and z_in_bounds)
