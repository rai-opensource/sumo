# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np
from judo.utils.fields import np_1d_field
from mujoco import MjData, MjModel

from sumo import MODEL_PATH
from sumo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from sumo.tasks.spot.spot_constants import (
    GRIPPER_CLOSED_POS,
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_barrier.xml")

Z_AXIS = np.array([0.0, 0.0, 1.0])
Y_AXIS = np.array([1.0, 0.0, 0.0])
X_AXIS = np.array([0.0, -1.0, 0.0])  # flipped for hardware

# annulus object position sampling
RADIUS_MIN = 1.5
RADIUS_MAX = 3.0
USE_LEGS = False
USE_GRIPPER = True
USE_TORSO = True

HARDWARE_FENCE_X = (-2.0, 3.0)
HARDWARE_FENCE_Y = (-3.0, 2.5)

DEFAULT_SPOT_POS = np.array([-1.5, 0.0])
DEFAULT_OBJECT_POS = np.array([2.0, 0.0])


@dataclass
class SpotBarrierDragConfig(SpotBaseConfig):
    """Config for the spot barrier dragging task."""

    goal_position: np.ndarray = np_1d_field(
        np.array([-0.0, 0, 0.0], dtype=np.float64),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, -1.0],
        maxs=[5.0, 5.0, 2.0],
        steps=[0.1, 0.1, 0.1],
        vis_name="barrier_goal_position",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[-1.0, 1.3, 0.0],
    )
    w_fence: float = 1000.0
    w_goal: float = 500.0
    w_orientation: float = 50.0  # Reward for keeping barrier aligned with global x, y, z axes
    w_heading_alignment: float = 500.0  # Reward for barrier facing towards goal
    object_fallen_threshold: float = 0.5
    orientation_threshold: float = 0.7
    w_torso_proximity: float = 50.0
    torso_proximity_threshold: float = 1.0
    w_object_fallen: float = 2500.0
    w_gripper_to_grasp_proximity: float = 250.0
    w_gripper_orientation: float = 100.0
    w_approach_site_proximity: float = 100.0
    w_controls: float = 2.0
    w_object_velocity: float = 10.0
    # Resistance-based grasp detection parameters
    resistance_threshold: float = 0.15
    fully_closed_tolerance: float = 0.05
    w_false_grasp_penalty: float = 1000.0
    w_grasp_quality: float = 100.0
    position_tolerance: float = 0.2


class SpotBarrierDrag(SpotBase[SpotBarrierDragConfig]):
    """Task getting Spot to drag an upright crowd barrier to a goal location."""

    name = "spot_barrier_drag"
    config_t = SpotBarrierDragConfig
    config: SpotBarrierDragConfig

    def __init__(self, config: SpotBarrierDragConfig | None = None) -> None:
        super().__init__(model_path=XML_PATH, use_arm=True, use_gripper=True, use_torso=True, config=config)

        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.object_pose_idx = self.get_joint_position_start_index("crowd_barrier_joint")
        self.object_vel_idx = self.get_joint_velocity_start_index("crowd_barrier_joint")
        self.object_x_axis_idx = self.get_sensor_start_index("object_x_axis")
        self.object_y_axis_idx = self.get_sensor_start_index("object_y_axis")
        self.object_z_axis_idx = self.get_sensor_start_index("object_z_axis")

        # Grasp point proximity sensors (left and right mid-height points for drag task)
        self.gripper_to_grasp_left_idx = self.get_sensor_start_index("sensor_gripper_to_grasp_drag_left")
        self.gripper_to_grasp_right_idx = self.get_sensor_start_index("sensor_gripper_to_grasp_drag_right")

        # Gripper orientation sensors
        self.gripper_x_axis_idx = self.get_sensor_start_index("sensor_gripper_x_axis")
        self.gripper_y_axis_idx = self.get_sensor_start_index("sensor_gripper_y_axis")
        self.gripper_z_axis_idx = self.get_sensor_start_index("sensor_gripper_z_axis")

        # Gripper position sensor
        self.gripper_pos_idx = self.get_sensor_start_index("trace_fngr_site")

        # Torso to approach site proximity sensors (left and right)
        self.torso_to_approach_left_idx = self.get_sensor_start_index("sensor_torso_to_approach_left")
        self.torso_to_approach_right_idx = self.get_sensor_start_index("sensor_torso_to_approach_right")

        # Get gripper joint position index for closure reward
        self.gripper_joint_idx = self.get_joint_position_start_index("arm_f1x")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the Spot crowd barrier dragging task."""
        batch_size = states.shape[0]

        qpos = states[..., : self.model.nq]

        body_height = qpos[..., self.body_pose_idx + 2]
        body_pos = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]
        object_pos = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        object_linear_velocity = states[..., self.object_vel_idx : self.object_vel_idx + 3]

        object_z_axis = sensors[..., self.object_z_axis_idx : self.object_z_axis_idx + 3]
        object_x_axis = sensors[..., self.object_x_axis_idx : self.object_x_axis_idx + 3]
        object_y_axis = sensors[..., self.object_y_axis_idx : self.object_y_axis_idx + 3]

        # Grasp point proximity vectors
        gripper_to_grasp_left = sensors[..., self.gripper_to_grasp_left_idx : self.gripper_to_grasp_left_idx + 3]
        gripper_to_grasp_right = sensors[..., self.gripper_to_grasp_right_idx : self.gripper_to_grasp_right_idx + 3]

        # Gripper orientation axes
        gripper_x_axis = sensors[..., self.gripper_x_axis_idx : self.gripper_x_axis_idx + 3]
        gripper_y_axis = sensors[..., self.gripper_y_axis_idx : self.gripper_y_axis_idx + 3]

        # Gripper position
        gripper_pos = sensors[..., self.gripper_pos_idx : self.gripper_pos_idx + 3]

        # Torso to approach site vectors
        torso_to_approach_left = sensors[..., self.torso_to_approach_left_idx : self.torso_to_approach_left_idx + 3]
        torso_to_approach_right = sensors[..., self.torso_to_approach_right_idx : self.torso_to_approach_right_idx + 3]

        # === Resistance-Based Grasp Detection ===
        gripper_joint_pos = qpos[..., self.gripper_joint_idx]
        gripper_joint_cmd = controls[..., 9]  # ARM_CMD_INDS[6] maps to controls index 9

        # Position error: actual - commanded
        position_error = gripper_joint_pos - gripper_joint_cmd

        # Grasp detection logic
        has_resistance = position_error > self.config.resistance_threshold
        is_fully_closed = np.abs(gripper_joint_pos - GRIPPER_CLOSED_POS) < self.config.fully_closed_tolerance
        is_not_empty = ~is_fully_closed
        is_grasping = has_resistance & is_not_empty
        is_false_grasp = is_fully_closed & (gripper_joint_cmd < -0.5)

        # Fence reward
        fence_violated_x = (body_pos[..., 0] < HARDWARE_FENCE_X[0]) | (body_pos[..., 0] > HARDWARE_FENCE_X[1])
        fence_violated_y = (body_pos[..., 1] < HARDWARE_FENCE_Y[0]) | (body_pos[..., 1] > HARDWARE_FENCE_Y[1])
        spot_fence_reward = -self.config.w_fence * (fence_violated_x | fence_violated_y).any(axis=-1)

        # Check if spot has fallen
        spot_fallen_reward = -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        # Check if object has fallen
        object_fallen_reward = -self.config.w_object_fallen * (
            object_pos[..., 2] <= self.config.object_fallen_threshold
        ).any(axis=-1)

        # Goal reward - distance to goal position
        goal_reward = -self.config.w_goal * np.linalg.norm(
            object_pos - np.array(self.config.goal_position)[None, None], axis=-1
        ).mean(-1)

        # Orientation reward - keep barrier aligned with global x, y, z axes
        x_alignment = np.sum(object_x_axis * X_AXIS, axis=-1)
        y_alignment = np.sum(object_y_axis * Y_AXIS, axis=-1)
        z_alignment = np.sum(object_z_axis * Z_AXIS, axis=-1)
        orientation_error = (1.0 - x_alignment) + (1.0 - y_alignment) + (1.0 - z_alignment)
        object_orientation_reward = -self.config.w_orientation * orientation_error.mean(axis=-1)

        # Heading alignment reward - keep barrier's x-axis pointing towards goal
        barrier_to_goal = np.array(self.config.goal_position)[None, None, :2] - object_pos[..., :2]
        barrier_to_goal_norm = barrier_to_goal / (np.linalg.norm(barrier_to_goal, axis=-1, keepdims=True) + 1e-8)

        object_x_horizontal = object_x_axis[..., :2]
        object_x_horizontal_norm = object_x_horizontal / (
            np.linalg.norm(object_x_horizontal, axis=-1, keepdims=True) + 1e-8
        )

        heading_alignment = np.sum(object_x_horizontal_norm * barrier_to_goal_norm, axis=-1)
        heading_alignment_abs = np.abs(heading_alignment)
        heading_error = 1.0 - heading_alignment_abs
        heading_alignment_reward = -self.config.w_heading_alignment * heading_error.mean(axis=-1)

        # Approach site proximity reward
        distance_to_left_approach = np.linalg.norm(torso_to_approach_left, axis=-1)
        distance_to_right_approach = np.linalg.norm(torso_to_approach_right, axis=-1)
        approach_distances = np.stack([distance_to_left_approach, distance_to_right_approach], axis=-1)
        min_approach_distance = np.min(approach_distances, axis=-1)
        approach_site_proximity_reward = -self.config.w_approach_site_proximity * min_approach_distance.mean(-1)

        # Torso proximity reward
        torso_proximity_reward = self.config.w_torso_proximity * np.minimum(
            self.config.torso_proximity_threshold, np.linalg.norm(body_pos - object_pos, axis=-1)
        ).mean(-1)

        # Gripper to grasp point proximity
        distance_to_left = np.linalg.norm(gripper_to_grasp_left, axis=-1)
        distance_to_right = np.linalg.norm(gripper_to_grasp_right, axis=-1)
        grasp_distances = np.stack([distance_to_left, distance_to_right], axis=-1)
        min_grasp_distance = np.min(grasp_distances, axis=-1)
        gripper_to_grasp_proximity_reward = -self.config.w_gripper_to_grasp_proximity * min_grasp_distance.mean(-1)

        # Gripper orientation reward - adaptive based on position
        object_to_gripper = gripper_pos - object_pos

        gripper_x_axis_norm = gripper_x_axis / (np.linalg.norm(gripper_x_axis, axis=-1, keepdims=True) + 1e-8)
        object_y_axis_norm = object_y_axis / (np.linalg.norm(object_y_axis, axis=-1, keepdims=True) + 1e-8)

        # Determine if gripper is in front (+Y) or behind (-Y) the object
        projection_on_y = np.sum(object_to_gripper * object_y_axis_norm, axis=-1, keepdims=True)

        # Gripper should point toward the object
        target_y_axis = np.where(projection_on_y > 0, -object_y_axis_norm, object_y_axis_norm)

        x_dot_product = np.sum(gripper_x_axis_norm * target_y_axis, axis=-1)
        x_alignment_error = 1.0 - x_dot_product

        # Gripper Y-axis aligned with object Z-axis (upward)
        gripper_y_axis_norm = gripper_y_axis / (np.linalg.norm(gripper_y_axis, axis=-1, keepdims=True) + 1e-8)
        object_z_axis_norm = object_z_axis / (np.linalg.norm(object_z_axis, axis=-1, keepdims=True) + 1e-8)
        y_dot_product = np.sum(gripper_y_axis_norm * object_z_axis_norm, axis=-1)
        y_alignment_error = 1.0 - y_dot_product

        total_orientation_error = x_alignment_error + y_alignment_error
        gripper_orientation_reward = -self.config.w_gripper_orientation * total_orientation_error.mean(axis=-1)

        # Object velocity penalty
        object_linear_velocity_reward = -self.config.w_object_velocity * np.square(
            np.linalg.norm(object_linear_velocity, axis=-1).mean(-1)
        )

        # Control penalty
        controls_reward = -self.config.w_controls * np.linalg.norm(controls[..., :3], axis=-1).mean(-1)

        # Grasp-based rewards
        false_grasp_penalty = -self.config.w_false_grasp_penalty * is_false_grasp.any(axis=-1)
        grasp_quality = np.clip(position_error / 0.5, 0, 1)
        grasp_quality_reward = self.config.w_grasp_quality * (is_grasping * grasp_quality).mean(axis=-1)

        assert spot_fence_reward.shape == (batch_size,)
        assert spot_fallen_reward.shape == (batch_size,)
        assert object_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert object_orientation_reward.shape == (batch_size,)
        assert heading_alignment_reward.shape == (batch_size,)
        assert approach_site_proximity_reward.shape == (batch_size,)
        assert torso_proximity_reward.shape == (batch_size,)
        assert gripper_to_grasp_proximity_reward.shape == (batch_size,)
        assert gripper_orientation_reward.shape == (batch_size,)
        assert object_linear_velocity_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)
        assert false_grasp_penalty.shape == (batch_size,)
        assert grasp_quality_reward.shape == (batch_size,)

        return (
            spot_fence_reward
            + spot_fallen_reward
            + object_fallen_reward
            + goal_reward
            + object_orientation_reward
            + heading_alignment_reward
            + gripper_to_grasp_proximity_reward
            + gripper_orientation_reward
            + false_grasp_penalty
            + grasp_quality_reward
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object - barrier starts upright."""
        # Barrier starts upright (no rotation)
        reset_object_pose = np.array([*DEFAULT_OBJECT_POS, 0.035, 1, 0, 0, 0])

        return np.array(
            [
                *DEFAULT_SPOT_POS,
                STANDING_HEIGHT,
                1,
                0,
                0,
                0,
                *LEGS_STANDING_POS,
                *self.reset_arm_pos,
                *reset_object_pose,
            ]
        )

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the barrier has been successfully dragged to the goal position."""
        object_pos = data.qpos[self.object_pose_idx : self.object_pose_idx + 3]
        goal_position = np.array(self.config.goal_position)

        # Check position tolerance
        distance_to_goal = np.linalg.norm(object_pos - goal_position)
        position_success = distance_to_goal <= self.config.position_tolerance

        return bool(position_success)
