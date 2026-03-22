# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""SpotChairPush task - push a yellow chair to a goal.

Adapted from judo/tasks/spot/spot_yellow_chair_push.py.
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

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_yellow_chair.xml")

RADIUS_MIN = 1.0
RADIUS_MAX = 2.0
HARDWARE_FENCE_X = (-2.0, 3.0)
HARDWARE_FENCE_Y = (-3.0, 2.5)


@dataclass
class SpotChairPushConfig(SpotBaseConfig):
    """Configuration for the SpotChairPush task."""

    w_fence: float = 1000.0
    w_goal: float = 60.0
    w_orientation: float = 50.0
    orientation_sparsity: float = 5.0
    w_torso_proximity: float = 50.0
    torso_proximity_threshold: float = 1.0
    w_gripper_proximity: float = 8.0
    orientation_threshold: float = 0.7
    w_controls: float = 2.0
    w_object_velocity: float = 64.0
    fall_penalty: float = 2500.0
    goal_position: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, 0.0]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, None],
    )

    # Success criteria (sumo-specific)
    orientation_tolerance: float = 0.1


class SpotChairPush(SpotBase[SpotChairPushConfig]):
    """Task getting Spot to push a yellow chair to a desired goal location."""

    name: str = "spot_chair_push"
    config_t: type[SpotChairPushConfig] = SpotChairPushConfig  # type: ignore[assignment]
    config: SpotChairPushConfig

    def __init__(
        self,
        config: SpotChairPushConfig | None = None,
    ) -> None:
        """Initialize the SpotChairPush task."""
        super().__init__(model_path=XML_PATH, use_arm=True, config=config)
        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.object_pose_idx = self.get_joint_position_start_index("yellow_chair_joint")
        self.gripper_pos_idx = self.get_sensor_start_index("trace_fngr_site")
        self.object_z_axis_idx = self.get_sensor_start_index("object_z_axis")
        self.object_vel_idx = self.model.jnt_dofadr[self.model.joint("yellow_chair_joint").id]

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the chair push task."""
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq :]

        body_height = qpos[..., self.body_pose_idx + 2]
        body_pos = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]
        object_pos = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        object_linear_velocity = qvel[..., self.object_vel_idx : self.object_vel_idx + 3]
        object_z_axis = sensors[..., self.object_z_axis_idx : self.object_z_axis_idx + 3]
        gripper_pos = sensors[..., self.gripper_pos_idx : self.gripper_pos_idx + 3]

        # Fence penalty
        fence_violated_x = (body_pos[..., 0] < HARDWARE_FENCE_X[0]) | (body_pos[..., 0] > HARDWARE_FENCE_X[1])
        fence_violated_y = (body_pos[..., 1] < HARDWARE_FENCE_Y[0]) | (body_pos[..., 1] > HARDWARE_FENCE_Y[1])
        spot_fence_reward = -self.config.w_fence * (fence_violated_x | fence_violated_y).any(axis=-1)

        spot_fallen_reward = -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        goal_reward = -self.config.w_goal * np.linalg.norm(
            object_pos - self.config.goal_position[None, None], axis=-1
        ).mean(-1)

        orientation_alignment = np.minimum(np.dot(object_z_axis, Z_AXIS) - 1, self.config.orientation_threshold)
        object_orientation_reward = +self.config.w_orientation * np.exp(
            self.config.orientation_sparsity * orientation_alignment
        ).sum(axis=-1)

        torso_proximity_reward = self.config.w_torso_proximity * np.minimum(
            self.config.torso_proximity_threshold, np.linalg.norm(body_pos - object_pos, axis=-1)
        ).mean(-1)

        gripper_proximity_reward = -self.config.w_gripper_proximity * np.linalg.norm(
            gripper_pos - object_pos, axis=-1
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
        """Reset pose for the chair push task."""
        radius = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        object_xy = np.array([radius * np.cos(theta), radius * np.sin(theta)]) + np.random.randn(2)
        reset_object_pose = np.array([*object_xy, 0, 1, 0, 0, 0])
        return np.array(
            [
                *np.random.randn(2),
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
        """Check if the yellow chair is upright, regardless of position."""
        object_z_axis = data.sensordata[self.object_z_axis_idx : self.object_z_axis_idx + 3]
        orientation_alignment = np.dot(object_z_axis, Z_AXIS)
        orientation_success = orientation_alignment >= (1.0 - self.config.orientation_tolerance)
        return bool(orientation_success)
