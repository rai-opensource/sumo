# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from judo.tasks.spot.spot_box_push import SpotBoxPush as _JudoSpotBoxPush
from judo.tasks.spot.spot_constants import BOX_HALF_LENGTH
from judo.utils.fields import np_1d_field

from sumo.tasks.spot.spot_base import SpotAssetMixin
from sumo.tasks.spot.spot_push import (
    SpotPushConfig,
    goal_distance_reward,
    gripper_distance_reward,
    object_linear_velocity_reward,
)


@dataclass
class SpotBoxPushConfig(SpotPushConfig):
    """Configuration for Sumo's simplified Spot box pushing analysis task."""

    goal_position: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, BOX_HALF_LENGTH]),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 3.0],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, None],
    )


class SpotBoxPush(SpotAssetMixin, _JudoSpotBoxPush):
    """Spot box pushing with Sumo's simplified analysis reward."""

    config_t: type[SpotBoxPushConfig] = SpotBoxPushConfig  # type: ignore[assignment]
    config: Any

    def __init__(self, config: SpotBoxPushConfig | None = None) -> None:
        super().__init__(config=cast(Any, config))
        self.object_vel_idx = self.get_joint_velocity_start_index("box_joint")

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
        gripper_pos = sensors[..., self.gripper_pos_idx : self.gripper_pos_idx + 3]
        object_linear_velocity = states[..., self.object_vel_idx : self.object_vel_idx + 3]

        goal_reward = goal_distance_reward(self.config, object_pos)
        gripper_proximity_reward = gripper_distance_reward(
            self.config, np.linalg.norm(gripper_pos - object_pos, axis=-1)
        )
        object_linear_velocity_penalty = object_linear_velocity_reward(self.config, object_linear_velocity)

        assert goal_reward.shape == (batch_size,)
        assert gripper_proximity_reward.shape == (batch_size,)
        assert object_linear_velocity_penalty.shape == (batch_size,)
        return goal_reward + gripper_proximity_reward + object_linear_velocity_penalty


__all__ = ["SpotBoxPush", "SpotBoxPushConfig"]
