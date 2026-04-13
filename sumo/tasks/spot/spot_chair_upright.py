# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np
from mujoco import MjData, MjModel

from sumo.tasks.spot.spot_chair_push import SpotChairPush
from sumo.tasks.spot.spot_constants import LEGS_STANDING_POS, STANDING_HEIGHT
from sumo.tasks.spot.spot_upright import (
    Z_AXIS,
    SpotUprightConfig,
    gripper_distance_reward,
    random_object_pose,
    sample_annulus_xy,
    z_axis_orientation_reward,
)

RADIUS_MIN = 1.0
RADIUS_MAX = 2.0
ORIENTATION_TOLERANCE = 0.1


@dataclass
class SpotChairUprightConfig(SpotUprightConfig):
    """Configuration for Sumo's simplified Spot chair upright analysis task."""


class SpotChairUpright(SpotChairPush):
    """Task getting Spot to upright a randomly oriented chair."""

    name: str = "spot_chair_upright"
    config_t: type[SpotChairUprightConfig] = SpotChairUprightConfig  # type: ignore[assignment]
    config: SpotChairUprightConfig

    def __init__(self, config: SpotChairUprightConfig | None = None) -> None:
        super().__init__(config=config)
        self.object_z_axis_idx = self.get_sensor_start_index("object_z_axis")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward using only object orientation and gripper distance."""
        batch_size = states.shape[0]
        qpos = states[..., : self.model.nq]

        object_pos = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        object_z_axis = sensors[..., self.object_z_axis_idx : self.object_z_axis_idx + 3]
        gripper_pos = sensors[..., self.gripper_pos_idx : self.gripper_pos_idx + 3]
        gripper_distance = np.linalg.norm(gripper_pos - object_pos, axis=-1)

        orientation_reward = z_axis_orientation_reward(self.config, object_z_axis)
        proximity_reward = gripper_distance_reward(self.config, gripper_distance)

        assert orientation_reward.shape == (batch_size,)
        assert proximity_reward.shape == (batch_size,)
        return orientation_reward + proximity_reward

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose with a random chair attitude that clears the ground."""
        object_pose = random_object_pose(self.model, "yellow_chair", sample_annulus_xy(RADIUS_MIN, RADIUS_MAX))
        robot_pose = np.array([0.0, 0.0, STANDING_HEIGHT, 1.0, 0.0, 0.0, 0.0])
        return np.array([*robot_pose, *LEGS_STANDING_POS, *self.reset_arm_pos, *object_pose])

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the chair is upright."""
        object_z_axis = data.sensordata[self.object_z_axis_idx : self.object_z_axis_idx + 3]
        return bool(np.dot(object_z_axis, Z_AXIS) >= 1.0 - ORIENTATION_TOLERANCE)


__all__ = ["SpotChairUpright", "SpotChairUprightConfig"]
