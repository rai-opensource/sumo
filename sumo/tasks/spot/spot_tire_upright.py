# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from mujoco import MjData, MjModel

from sumo.tasks.spot.spot_constants import LEGS_STANDING_POS, STANDING_HEIGHT
from sumo.tasks.spot.spot_tire_push import RADIUS_MAX, RADIUS_MIN, SpotTirePush
from sumo.tasks.spot.spot_upright import (
    SpotUprightConfig,
    gripper_distance_reward,
    horizontal_axis_orientation_reward,
    random_object_pose,
    sample_annulus_xy,
    y_axis_is_horizontal,
)

ORIENTATION_TOLERANCE = 0.1


@dataclass
class SpotTireUprightConfig(SpotUprightConfig):
    """Configuration for Sumo's simplified Spot tire upright analysis task."""


class SpotTireUpright(SpotTirePush):
    """Task getting Spot to upright a randomly oriented tire."""

    name: str = "spot_tire_upright"
    config_t: type[SpotTireUprightConfig] = SpotTireUprightConfig  # type: ignore[assignment]
    config: SpotTireUprightConfig

    def __init__(self, config: SpotTireUprightConfig | None = None) -> None:
        super().__init__(config=cast(Any, config))
        self.object_y_axis_idx = self.get_sensor_start_index("object_y_axis")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward using only tire orientation and gripper distance."""
        batch_size = states.shape[0]

        tire_y_axis = sensors[..., self.object_y_axis_idx : self.object_y_axis_idx + 3]
        gripper_to_object = sensors[..., self.end_effector_to_object_start : self.end_effector_to_object_start + 3]
        gripper_distance = np.linalg.norm(gripper_to_object, axis=-1)

        orientation_reward = horizontal_axis_orientation_reward(self.config, tire_y_axis)
        proximity_reward = gripper_distance_reward(self.config, gripper_distance)

        assert orientation_reward.shape == (batch_size,)
        assert proximity_reward.shape == (batch_size,)
        return orientation_reward + proximity_reward

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose with a random tire attitude that clears the ground."""
        object_pose = random_object_pose(
            self.model,
            "tire",
            sample_annulus_xy(RADIUS_MIN, RADIUS_MAX),
            reject_orientation=lambda quat: y_axis_is_horizontal(quat, ORIENTATION_TOLERANCE),
        )
        robot_pose = np.array([0.0, 0.0, STANDING_HEIGHT, 1.0, 0.0, 0.0, 0.0])
        return np.array([*robot_pose, *LEGS_STANDING_POS, *self.reset_arm_pos, *object_pose])

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the tire is upright."""
        tire_y_axis = data.sensordata[self.object_y_axis_idx : self.object_y_axis_idx + 3]
        return bool(abs(tire_y_axis[2]) <= ORIENTATION_TOLERANCE)


__all__ = ["SpotTireUpright", "SpotTireUprightConfig"]
