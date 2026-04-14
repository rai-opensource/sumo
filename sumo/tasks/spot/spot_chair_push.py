# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""SpotChairPush task - push a yellow chair to a goal.

Adapted from judo/tasks/spot/spot_yellow_chair_push.py.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from judo.tasks.base import TaskConfig
from judo.utils.fields import np_1d_field
from mujoco import MjData, MjModel

from sumo import MODEL_PATH
from sumo.tasks.spot.spot_base import SpotBase
from sumo.tasks.spot.spot_constants import LEGS_STANDING_POS, STANDING_HEIGHT

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_yellow_chair.xml")

RADIUS_MIN = 1.0
RADIUS_MAX = 2.0
POSITION_TOLERANCE = 0.2
VELOCITY_TOLERANCE = 0.05


@dataclass
class SpotChairPushConfig(TaskConfig):
    """Configuration for Sumo's simplified Spot chair pushing analysis task."""

    w_goal: float = 60.0
    w_gripper_proximity: float = 4.0
    w_object_velocity: float = 20.0
    goal_position: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, 0.0]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, None],
    )


class SpotChairPush(SpotBase):
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
        self.object_vel_idx = self.get_joint_velocity_start_index("yellow_chair_joint")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward using only goal distance, gripper distance, and object velocity."""
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]

        object_pos = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        object_linear_velocity = states[..., self.object_vel_idx : self.object_vel_idx + 3]
        gripper_pos = sensors[..., self.gripper_pos_idx : self.gripper_pos_idx + 3]

        goal_reward = -self.config.w_goal * np.linalg.norm(
            object_pos - self.config.goal_position[None, None], axis=-1
        ).mean(-1)

        gripper_proximity_reward = -self.config.w_gripper_proximity * np.linalg.norm(
            gripper_pos - object_pos, axis=-1
        ).mean(-1)

        object_linear_velocity_reward = -self.config.w_object_velocity * np.square(
            np.linalg.norm(object_linear_velocity, axis=-1).mean(-1)
        )

        assert goal_reward.shape == (batch_size,)
        assert gripper_proximity_reward.shape == (batch_size,)
        assert object_linear_velocity_reward.shape == (batch_size,)
        return goal_reward + gripper_proximity_reward + object_linear_velocity_reward

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
        """Check if the yellow chair is in the goal position."""
        object_pos = data.qpos[self.object_pose_idx : self.object_pose_idx + 3]
        object_vel = data.qvel[self.object_vel_idx - self.model.nq : self.object_vel_idx - self.model.nq + 3]
        position_check = (
            np.linalg.norm(object_pos - self.config.goal_position, axis=-1, ord=np.inf) < POSITION_TOLERANCE
        )
        velocity_check = np.linalg.norm(object_vel, axis=-1) < VELOCITY_TOLERANCE
        return bool(position_check and velocity_check)
