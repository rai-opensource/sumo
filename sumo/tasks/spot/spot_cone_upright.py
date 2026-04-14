# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""SpotConeUpright task - upright a fallen traffic cone.

The robot must pick up or nudge the cone back to an upright position.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from mujoco import MjData, MjModel

from sumo import MODEL_PATH
from sumo.tasks.spot.spot_base import SpotBase
from sumo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)
from sumo.tasks.spot.spot_upright import (
    Z_AXIS,
    SpotUprightConfig,
    gripper_distance_reward,
    random_object_pose,
    sample_annulus_xy,
    z_axis_orientation_reward,
)

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_traffic_cone.xml")

RADIUS_MIN = 0.2
RADIUS_MAX = 0.5

DEFAULT_SPOT_POS = np.array([-1.5, 0.0])
ORIENTATION_TOLERANCE = 0.1


@dataclass
class SpotConeUprightConfig(SpotUprightConfig):
    """Config for Sumo's simplified Spot cone upright analysis task."""


class SpotConeUpright(SpotBase[SpotConeUprightConfig]):
    """Task getting Spot to upright a fallen traffic cone."""

    name: str = "spot_cone_upright"
    config_t: type[SpotConeUprightConfig] = SpotConeUprightConfig  # type: ignore[assignment]
    config: SpotConeUprightConfig

    def __init__(self, config: SpotConeUprightConfig | None = None) -> None:
        super().__init__(model_path=XML_PATH, use_arm=True, config=config)

        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.object_pose_idx = self.get_joint_position_start_index("traffic_cone_joint")
        self.object_z_axis_idx = self.get_sensor_start_index("object_z_axis")
        self.end_effector_to_object_idx = self.get_sensor_start_index("sensor_arm_link_fngr")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward using only object orientation and gripper distance."""
        batch_size = states.shape[0]

        object_z_axis = sensors[..., self.object_z_axis_idx : self.object_z_axis_idx + 3]
        gripper_to_object = sensors[..., self.end_effector_to_object_idx : self.end_effector_to_object_idx + 3]
        gripper_distance = np.linalg.norm(gripper_to_object, axis=-1)

        orientation_reward = z_axis_orientation_reward(self.config, object_z_axis)
        proximity_reward = gripper_distance_reward(self.config, gripper_distance)

        assert orientation_reward.shape == (batch_size,)
        assert proximity_reward.shape == (batch_size,)
        return orientation_reward + proximity_reward

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose with a random cone attitude that clears the ground."""
        reset_object_pose = random_object_pose(
            self.model,
            "traffic_cone",
            sample_annulus_xy(RADIUS_MIN, RADIUS_MAX),
        )
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
        return bool(orientation_alignment >= (1.0 - ORIENTATION_TOLERANCE))


__all__ = ["SpotConeUpright", "SpotConeUprightConfig"]
