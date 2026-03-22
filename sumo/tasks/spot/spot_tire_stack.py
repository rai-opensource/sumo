# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""SpotTireStack task - stack one tire on another.

Adapted from judo/tasks/spot/spot_tire_stack.py.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from judo.tasks.spot.spot_utils import apply_quat_to_vec
from mujoco import MjData, MjModel

from sumo import MODEL_PATH
from sumo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from sumo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
    TIRE_HALF_WIDTH,
    TIRE_RADIUS,
)

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_two_tires.xml")


@dataclass
class SpotTireStackConfig(SpotBaseConfig):
    """Configuration for the SpotTireStack task."""

    w_tire_stack_xy: float = 100.0
    w_tire_stack_z: float = 200.0
    w_bottom_tire_velocity: float = 50.0
    w_gripper_proximity: float = 10.0
    w_torso_proximity: float = 10.0
    w_tire_orientation: float = 20.0
    fall_penalty: float = 10_000.0
    w_controls: float = 1.0
    w_gripper_height_penalty: float = 1000.0
    orientation_reward_distance_threshold: float = 2.0

    # Success criteria (sumo-specific)
    stack_xy_tolerance: float = 0.1
    stack_z_tolerance: float = 0.1


class SpotTireStack(SpotBase[SpotTireStackConfig]):
    """Task getting Spot to stack one tire on top of another."""

    name: str = "spot_tire_stack"
    config_t: type[SpotTireStackConfig] = SpotTireStackConfig  # type: ignore[assignment]
    config: SpotTireStackConfig

    def __init__(
        self,
        config: SpotTireStackConfig | None = None,
    ) -> None:
        """Initialize the SpotTireStack task."""
        super().__init__(model_path=XML_PATH, use_arm=True, use_torso=False, config=config)
        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.bottom_tire_pose_idx = self.get_joint_position_start_index("tire_rubber_joint")
        self.top_tire_pose_idx = self.get_joint_position_start_index("tire_rubber_2_joint")
        self.gripper_pos_idx = self.get_sensor_start_index("trace_fngr_site")
        self.top_tire_y_axis_idx = self.get_sensor_start_index("top_tire_y_axis")

        # Velocity indices
        self.bottom_tire_vel_idx = self.model.jnt_dofadr[self.model.joint("tire_rubber_joint").id]
        self.top_tire_vel_idx = self.model.jnt_dofadr[self.model.joint("tire_rubber_2_joint").id]

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the tire stack task."""
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq :]

        p_W_bottom_tire = qpos[..., self.bottom_tire_pose_idx : self.bottom_tire_pose_idx + 3]
        p_W_top_tire = qpos[..., self.top_tire_pose_idx : self.top_tire_pose_idx + 3]
        p_WB = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]
        p_WG = sensors[..., self.gripper_pos_idx : self.gripper_pos_idx + 3]

        # XY alignment
        p_bottom_to_top_xy = p_W_top_tire[..., :2] - p_W_bottom_tire[..., :2]
        xy_distance = np.linalg.norm(p_bottom_to_top_xy, axis=-1)
        tire_stack_xy_reward = -self.config.w_tire_stack_xy * xy_distance.mean(axis=-1)

        # Z stacking
        desired_top_tire_z = p_W_bottom_tire[..., 2] + TIRE_HALF_WIDTH * 2
        z_error = np.abs(p_W_top_tire[..., 2] - desired_top_tire_z)
        tire_stack_z_reward = -self.config.w_tire_stack_z * z_error.mean(axis=-1)

        # Tire orientation
        tire_distance = np.linalg.norm(p_W_top_tire - p_W_bottom_tire, axis=-1)
        orientation_active = tire_distance < self.config.orientation_reward_distance_threshold
        top_tire_y_axis = sensors[..., self.top_tire_y_axis_idx : self.top_tire_y_axis_idx + 3]
        top_to_bottom_vec = p_W_bottom_tire - p_W_top_tire
        top_to_bottom_norm = np.maximum(np.linalg.norm(top_to_bottom_vec, axis=-1, keepdims=True), 1e-6)
        top_to_bottom_unit = top_to_bottom_vec / top_to_bottom_norm
        dot_product = np.sum(top_tire_y_axis * top_to_bottom_unit, axis=-1)
        orientation_error = 1.0 - dot_product
        orientation_active_any = orientation_active.any(axis=-1)
        tire_orientation_reward = np.where(
            orientation_active_any, -self.config.w_tire_orientation * orientation_error.mean(axis=-1), 0.0
        )

        # Bottom tire velocity
        bottom_tire_lin_vel = qvel[..., self.bottom_tire_vel_idx : self.bottom_tire_vel_idx + 3]
        bottom_tire_ang_vel = qvel[..., self.bottom_tire_vel_idx + 3 : self.bottom_tire_vel_idx + 6]
        bottom_tire_velocity_reward = -self.config.w_bottom_tire_velocity * (
            np.linalg.norm(bottom_tire_lin_vel, axis=-1).mean(axis=-1)
            + np.linalg.norm(bottom_tire_ang_vel, axis=-1).mean(axis=-1)
        )

        # Torso shaping
        p_bottom_to_top = p_W_top_tire - p_W_bottom_tire
        p_bottom_to_top_norm = np.maximum(np.linalg.norm(p_bottom_to_top, axis=-1, keepdims=True), 1e-6)
        u_stacking_direction = p_bottom_to_top / p_bottom_to_top_norm

        max_distance = 3.0 * TIRE_RADIUS
        min_distance = 1.0 * TIRE_RADIUS
        normalized_distance = np.clip((tire_distance - min_distance) / (max_distance - min_distance), 0, 1)
        angle_degrees = -90.0 * (1.0 - normalized_distance) - 25.0
        angle_radians = np.deg2rad(angle_degrees)
        cos_half_angle = np.cos(angle_radians / 2)
        sin_half_angle = np.sin(angle_radians / 2)
        rotation_quat = np.stack(
            [cos_half_angle, np.zeros_like(cos_half_angle), np.zeros_like(cos_half_angle), sin_half_angle], axis=-1
        )
        u_stacking_direction_rotated = apply_quat_to_vec(rotation_quat, u_stacking_direction)

        distance_from_tire = 0.8
        p_WB_desired = p_W_top_tire + distance_from_tire * u_stacking_direction_rotated
        p_WB_desired[..., 2] = STANDING_HEIGHT

        torso_proximity_reward = -self.config.w_torso_proximity * np.linalg.norm(p_WB - p_WB_desired, axis=-1).mean(
            axis=-1
        )

        # Fall penalty
        body_height = qpos[..., self.body_pose_idx + 2]
        spot_fallen_reward = -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        # Gripper shaping
        z_world = np.zeros_like(p_W_top_tire)
        z_world[..., 2] = 1.0
        Ty_W = sensors[..., self.top_tire_y_axis_idx : self.top_tire_y_axis_idx + 3]
        z_proj_perp = z_world - np.sum(z_world * Ty_W, axis=-1, keepdims=True) * Ty_W
        z_proj_perp_norm = np.maximum(np.linalg.norm(z_proj_perp, axis=-1, keepdims=True), 1e-6)
        u_top = z_proj_perp / z_proj_perp_norm
        p_W_gripper_des = p_W_top_tire + TIRE_RADIUS * u_top
        gripper_proximity_reward = -self.config.w_gripper_proximity * np.linalg.norm(
            p_WG - p_W_gripper_des, axis=-1
        ).mean(axis=-1)

        # Gripper height penalty
        max_tire_height = 2 * TIRE_RADIUS
        gripper_height = p_WG[..., 2]
        gripper_height_excess = np.maximum(0, gripper_height - max_tire_height)
        gripper_height_penalty = -self.config.w_gripper_height_penalty * gripper_height_excess.mean(axis=-1)

        controls_reward = -self.config.w_controls * np.linalg.norm(controls, axis=-1).mean(-1)

        assert tire_stack_xy_reward.shape == (batch_size,)
        return (
            tire_stack_xy_reward
            + tire_stack_z_reward
            + tire_orientation_reward
            + bottom_tire_velocity_reward
            + gripper_proximity_reward
            + torso_proximity_reward
            + spot_fallen_reward
            + gripper_height_penalty
            + controls_reward
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose for the tire stack task."""
        robot_pose = np.array([-2.0, -2.0, STANDING_HEIGHT, 1.0, 0.0, 0.0, 0.0])

        # Bottom tire - lying flat
        tire_1_pose = np.zeros(7)
        tire_1_pose[0] = -2.0
        tire_1_pose[1] = 1.0
        tire_1_pose[2] = TIRE_HALF_WIDTH
        tire_1_pose[3:] = [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0]

        # Top tire - upright
        tire_2_pose = np.zeros(7)
        tire_2_pose[0] = -2.0
        tire_2_pose[1] = 0.0
        tire_2_pose[2] = TIRE_RADIUS
        tire_2_pose[3:] = [1.0, 0.0, 0.0, 0.0]

        return np.array([*robot_pose, *LEGS_STANDING_POS, *self.reset_arm_pos, *tire_1_pose, *tire_2_pose])

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the tires are successfully stacked."""
        bottom_tire_pos = data.qpos[self.bottom_tire_pose_idx : self.bottom_tire_pose_idx + 3]
        top_tire_pos = data.qpos[self.top_tire_pose_idx : self.top_tire_pose_idx + 3]

        xy_distance = np.linalg.norm(top_tire_pos[:2] - bottom_tire_pos[:2])
        xy_aligned = xy_distance <= self.config.stack_xy_tolerance

        desired_top_tire_z = bottom_tire_pos[2] + TIRE_HALF_WIDTH * 2
        z_error = np.abs(top_tire_pos[2] - desired_top_tire_z)
        z_correct = z_error <= self.config.stack_z_tolerance

        return bool(xy_aligned and z_correct)
