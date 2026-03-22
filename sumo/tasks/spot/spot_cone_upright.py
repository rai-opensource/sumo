# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""SpotConeUpright task - upright a fallen traffic cone.

The robot must pick up or nudge the cone back to an upright position.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from mujoco import MjData, MjModel

from sumo import MODEL_PATH
from sumo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from sumo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_traffic_cone.xml")

Z_AXIS = np.array([0.0, 0.0, 1.0])
RADIUS_MIN = 0.2
RADIUS_MAX = 0.5

HARDWARE_FENCE_X = (-2.0, 3.0)
HARDWARE_FENCE_Y = (-3.0, 2.5)

DEFAULT_SPOT_POS = np.array([-1.5, 0.0])
DEFAULT_OBJECT_POS = np.array([0.0, 0.0])


@dataclass
class SpotConeUprightConfig(SpotBaseConfig):
    """Config for the spot cone upright task."""

    goal_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    w_fence: float = 1000.0
    w_orientation: float = 50.0
    orientation_sparsity: float = 5.0
    w_torso_proximity: float = 50.0
    torso_proximity_threshold: float = 1.0
    w_gripper_proximity: float = 8.0
    orientation_threshold: float = 0.7
    w_controls: float = 2.0
    w_object_velocity: float = 64.0
    position_tolerance: float = 0.2
    orientation_tolerance: float = 0.1
    velocity_tolerance: float = 0.1


class SpotConeUpright(SpotBase[SpotConeUprightConfig]):
    """Task getting Spot to upright a fallen traffic cone."""

    name: str = "spot_cone_upright"
    config_t: type[SpotConeUprightConfig] = SpotConeUprightConfig  # type: ignore[assignment]
    config: SpotConeUprightConfig

    def __init__(self, config: SpotConeUprightConfig | None = None) -> None:
        super().__init__(model_path=XML_PATH, use_arm=True, config=config)

        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.object_pose_idx = self.get_joint_position_start_index("traffic_cone_joint")
        self.object_vel_idx = self.get_joint_velocity_start_index("traffic_cone_joint")
        self.object_y_axis_idx = self.get_sensor_start_index("object_y_axis")
        self.object_z_axis_idx = self.get_sensor_start_index("object_z_axis")
        self.end_effector_to_object_idx = self.get_sensor_start_index("sensor_arm_link_fngr")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the cone upright task."""
        batch_size = states.shape[0]

        qpos = states[..., : self.model.nq]

        body_height = qpos[..., self.body_pose_idx + 2]
        body_pos = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]
        object_pos = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        object_linear_velocity = states[..., self.object_vel_idx : self.object_vel_idx + 3]

        object_z_axis = sensors[..., self.object_z_axis_idx : self.object_z_axis_idx + 3]
        gripper_to_object = sensors[..., self.end_effector_to_object_idx : self.end_effector_to_object_idx + 3]

        fence_violated_x = (body_pos[..., 0] < HARDWARE_FENCE_X[0]) | (body_pos[..., 0] > HARDWARE_FENCE_X[1])
        fence_violated_y = (body_pos[..., 1] < HARDWARE_FENCE_Y[0]) | (body_pos[..., 1] > HARDWARE_FENCE_Y[1])
        spot_fence_reward = -self.config.w_fence * (fence_violated_x | fence_violated_y).any(axis=-1)

        spot_fallen_reward = -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        goal_reward = -self.config.w_goal * np.linalg.norm(
            object_pos - np.array(self.config.goal_position)[None, None], axis=-1
        ).mean(-1)

        orientation_alignment = np.minimum(np.dot(object_z_axis, Z_AXIS) - 1, self.config.orientation_threshold)
        object_orientation_reward = +self.config.w_orientation * np.exp(
            self.config.orientation_sparsity * orientation_alignment
        ).sum(axis=-1)

        torso_proximity_reward = self.config.w_torso_proximity * np.minimum(
            self.config.torso_proximity_threshold, np.linalg.norm(body_pos - object_pos, axis=-1)
        ).mean(-1)

        gripper_proximity_reward = -self.config.w_gripper_proximity * np.linalg.norm(
            gripper_to_object,
            axis=-1,
        ).mean(-1)

        object_linear_velocity_reward = -self.config.w_object_velocity * np.square(
            np.linalg.norm(object_linear_velocity, axis=-1).mean(-1)
        )

        controls_reward = -self.config.w_controls * np.linalg.norm(controls[..., :3], axis=-1).mean(-1)

        assert spot_fence_reward.shape == (batch_size,)
        return (
            spot_fence_reward
            + spot_fallen_reward
            + goal_reward
            + object_orientation_reward
            + torso_proximity_reward
            + gripper_proximity_reward
            + object_linear_velocity_reward
            + controls_reward
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose — cone starts fallen on its side."""
        radius = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        object_pos = np.array([radius * np.cos(theta), radius * np.sin(theta)]) + np.random.randn(2)
        # Cone on its side: 90-degree roll about x-axis
        reset_object_pose = np.array([*object_pos, 0.275, np.cos(np.pi / 4), -np.sin(np.pi / 4), 0, 0])
        spot_pos = DEFAULT_SPOT_POS + np.random.randn(2) * 0.001
        return np.array(
            [
                *spot_pos,
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
        """Check if the traffic cone is upright."""
        object_z_axis = data.sensordata[self.object_z_axis_idx : self.object_z_axis_idx + 3]
        orientation_alignment = np.dot(object_z_axis, Z_AXIS)
        orientation_success = orientation_alignment >= (1.0 - self.config.orientation_tolerance)

        vel_offset = self.object_vel_idx - self.model.nq
        velocity_success = (
            np.linalg.norm(data.qvel[vel_offset : vel_offset + 3], axis=-1) < self.config.velocity_tolerance
        )
        return bool(orientation_success and velocity_success)
