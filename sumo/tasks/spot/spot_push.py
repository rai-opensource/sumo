# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from judo.tasks.base import TaskConfig


@dataclass
class SpotPushConfig(TaskConfig):
    """Configuration for Sumo's simplified Spot pushing analysis tasks."""

    w_goal: float = 60.0
    w_gripper_proximity: float = 4.0
    w_object_velocity: float = 20.0


class SpotPushRewardConfig(Protocol):
    w_goal: float
    w_gripper_proximity: float
    w_object_velocity: float
    goal_position: np.ndarray


def goal_distance_reward(config: SpotPushRewardConfig, object_pos: np.ndarray) -> np.ndarray:
    """Reward object proximity to the goal position."""
    return -config.w_goal * np.linalg.norm(
        object_pos - np.asarray(config.goal_position)[None, None],
        axis=-1,
    ).mean(axis=-1)


def gripper_distance_reward(config: SpotPushRewardConfig, gripper_distance: np.ndarray) -> np.ndarray:
    """Reward gripper proximity to the pushed object."""
    return -config.w_gripper_proximity * gripper_distance.mean(axis=-1)


def object_linear_velocity_reward(config: SpotPushRewardConfig, object_linear_velocity: np.ndarray) -> np.ndarray:
    """Penalize object linear velocity."""
    return -config.w_object_velocity * np.square(np.linalg.norm(object_linear_velocity, axis=-1).mean(axis=-1))
