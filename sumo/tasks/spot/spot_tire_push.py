# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np
from judo.tasks.base import TaskConfig
from judo.utils.fields import np_1d_field
from mujoco import MjData, MjModel

from sumo import MODEL_PATH
from sumo.tasks.spot.spot_base import SpotBase
from sumo.tasks.spot.spot_constants import (
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_tire.xml")

TIRE_RADIUS: float = 0.339
TIRE_WIDTH: float = 0.175
RADIUS_MIN = 1.0
RADIUS_MAX = 2.0

# Success condition tolerances
POSITION_TOLERANCE = 0.1
VELOCITY_TOLERANCE = 0.05
SPOT_FALLEN_THRESHOLD = 0.35


@dataclass
class SpotTirePushConfig(TaskConfig):
    """Config for Sumo's simplified Spot tire pushing analysis task."""

    w_goal: float = 60.0
    w_gripper_proximity: float = 4.0
    w_object_velocity: float = 20.0

    goal_position: np.ndarray = np_1d_field(
        np.array([0.0, 0.0, TIRE_RADIUS], dtype=np.float64),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 1.0],
        steps=[0.1, 0.1, 0.05],
        vis_name="box_goal_position",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[0.0, 0.0, TIRE_RADIUS],
    )


class SpotTirePush(SpotBase[SpotTirePushConfig]):
    """Task getting Spot to push a tire to a goal location."""

    name = "spot_tire_push"
    config_t: type[SpotTirePushConfig] = SpotTirePushConfig
    config: Any

    def __init__(self, config: SpotTirePushConfig | None = None) -> None:
        super().__init__(model_path=XML_PATH, use_arm=True, config=config)

        self.body_pose_start = self.get_joint_position_start_index("base")
        self.object_pose_start = self.get_joint_position_start_index("tire_joint")
        self.object_vel_start = self.get_joint_velocity_start_index("tire_joint")
        self.end_effector_to_object_start = self.get_sensor_start_index("sensor_arm_link_fngr")

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
        object_pos = qpos[..., self.object_pose_start : self.object_pose_start + 3]
        object_linear_velocity = states[..., self.object_vel_start : self.object_vel_start + 3]

        end_effector_to_object = sensors[..., self.end_effector_to_object_start : self.end_effector_to_object_start + 3]
        gripper_proximity_reward = -self.config.w_gripper_proximity * np.linalg.norm(
            end_effector_to_object, axis=-1
        ).mean(axis=-1)

        goal_reward = -self.config.w_goal * np.linalg.norm(
            object_pos - np.array(self.config.goal_position)[None, None], axis=-1
        ).mean(-1)

        object_linear_velocity_penalty = -self.config.w_object_velocity * np.square(
            np.linalg.norm(object_linear_velocity, axis=-1).mean(-1)
        )

        assert gripper_proximity_reward.shape == (batch_size,)
        assert object_linear_velocity_penalty.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        return goal_reward + gripper_proximity_reward + object_linear_velocity_penalty

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object."""
        # Sample object position in annulus
        radius = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        object_pos = np.array([radius * np.cos(theta), radius * np.sin(theta)]) + 0.1 * np.random.randn(2)

        object_pose = np.array([*object_pos, TIRE_RADIUS, 1, 0, 0, 0])

        # Place robot at random x and y
        robot_pose_xy = np.random.uniform(-0.5, 0.5, 2)
        random_yaw_robot = np.random.uniform(0, 2 * np.pi)
        robot_pose_orientation = np.array([np.cos(random_yaw_robot / 2), 0, 0, np.sin(random_yaw_robot / 2)])
        robot_pose = np.array([*robot_pose_xy, STANDING_HEIGHT, *robot_pose_orientation])

        return np.array([*robot_pose, *LEGS_STANDING_POS, *self.reset_arm_pos, *object_pose])

    def success(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if the tire is in the goal position."""
        object_pos = data.qpos[self.object_pose_start : self.object_pose_start + 3]
        object_vel = data.qvel[self.object_vel_start - self.model.nq : self.object_vel_start - self.model.nq + 3]
        goal_pos = np.array(self.config.goal_position)
        position_check = np.linalg.norm(object_pos - goal_pos, axis=-1, ord=np.inf) < POSITION_TOLERANCE
        velocity_check = np.linalg.norm(object_vel, axis=-1) < VELOCITY_TOLERANCE
        return position_check and velocity_check

    def failure(self, model: MjModel, data: MjData, metadata: dict[str, Any] | None = None) -> bool:
        """Check if Spot has fallen."""
        body_height = data.qpos[self.body_pose_start + 2]
        return body_height <= SPOT_FALLEN_THRESHOLD
