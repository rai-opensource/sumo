# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np
from judo.utils.fields import np_1d_field
from mujoco import MjData, MjModel

from sumo import MODEL_PATH
from sumo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from sumo.tasks.spot.spot_constants import (
    GRIPPER_CLOSED_POS,
    LEGS_STANDING_POS,
    STANDING_HEIGHT,
)

XML_PATH = str(MODEL_PATH / "xml/spot_tasks/spot_barrier.xml")

Z_AXIS = np.array([0.0, 0.0, 1.0])
# annulus object position sampling
RADIUS_MIN = 1.5
RADIUS_MAX = 3.0
USE_LEGS = False
USE_GRIPPER = True
USE_TORSO = True

HARDWARE_FENCE_X = (-2.0, 3.0)
HARDWARE_FENCE_Y = (-3.0, 2.5)

DEFAULT_SPOT_POS = np.array([0.0, 0.0])
DEFAULT_OBJECT_POS = np.array([2.0, 0.0])


@dataclass
class SpotBarrierUprightConfig(SpotBaseConfig):
    """Config for the spot barrier uprighting task."""

    goal_position: np.ndarray = np_1d_field(
        np.array([-1.0, 1.3, 0.0], dtype=np.float64),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, -1.0],
        maxs=[5.0, 5.0, 2.0],
        steps=[0.1, 0.1, 0.1],
        vis_name="barrier_goal_position",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[-1.0, 1.3, 0.0],
    )
    w_goal: float = 0.0  # Override base class default
    w_fence: float = 1000.0
    w_orientation: float = 100.0
    orientation_sparsity: float = 3.0
    orientation_threshold: float = 0.7
    w_gripper_to_grasp_proximity: float = 500.0  # Reward for gripper proximity to grasp points
    w_gripper_orientation: float = 100.0
    w_controls: float = 2.0
    w_object_velocity: float = 10.0
    w_approach_site_proximity: float = 10.0  # Reward for torso proximity to approach sites
    # Resistance-based grasp detection parameters
    resistance_threshold: float = 0.15  # Position error indicating resistance (radians)
    fully_closed_tolerance: float = 0.05  # Tolerance for "fully closed" detection
    w_false_grasp_penalty: float = 1000.0  # Penalty for false grasp (closed but not grasping)
    w_grasp_quality: float = 100.0  # Reward for high-quality grasp
    position_tolerance: float = 0.2
    orientation_tolerance: float = 0.1


class SpotBarrierUpright(SpotBase[SpotBarrierUprightConfig]):
    """Task getting Spot to upright a fallen crowd barrier."""

    name = "spot_barrier_upright"
    config_t = SpotBarrierUprightConfig
    config: SpotBarrierUprightConfig

    def __init__(self, config: SpotBarrierUprightConfig | None = None) -> None:
        super().__init__(model_path=XML_PATH, use_arm=True, use_gripper=True, use_torso=True, config=config)

        self.body_pose_idx = self.get_joint_position_start_index("base")
        self.object_pose_idx = self.get_joint_position_start_index("crowd_barrier_joint")
        self.object_vel_idx = self.get_joint_velocity_start_index("crowd_barrier_joint")
        self.object_x_axis_idx = self.get_sensor_start_index("object_x_axis")
        self.object_y_axis_idx = self.get_sensor_start_index("object_y_axis")
        self.object_z_axis_idx = self.get_sensor_start_index("object_z_axis")

        # Grasp point proximity sensors (left and right mid-height points)
        self.gripper_to_grasp_left_idx = self.get_sensor_start_index("sensor_gripper_to_grasp_left")
        self.gripper_to_grasp_right_idx = self.get_sensor_start_index("sensor_gripper_to_grasp_right")

        # Gripper orientation sensors
        self.gripper_x_axis_idx = self.get_sensor_start_index("sensor_gripper_x_axis")
        self.gripper_y_axis_idx = self.get_sensor_start_index("sensor_gripper_y_axis")
        self.gripper_z_axis_idx = self.get_sensor_start_index("sensor_gripper_z_axis")

        # Torso to approach site proximity sensors (left and right)
        self.torso_to_approach_left_idx = self.get_sensor_start_index("sensor_torso_to_approach_left")
        self.torso_to_approach_right_idx = self.get_sensor_start_index("sensor_torso_to_approach_right")

        # Get gripper joint position index for closure reward
        self.gripper_joint_idx = self.get_joint_position_start_index("arm_f1x")

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the Spot crowd barrier uprighting task."""
        batch_size = states.shape[0]

        qpos = states[..., : self.model.nq]

        body_height = qpos[..., self.body_pose_idx + 2]
        body_pos = qpos[..., self.body_pose_idx : self.body_pose_idx + 3]
        object_pos = qpos[..., self.object_pose_idx : self.object_pose_idx + 3]
        object_linear_velocity = states[..., self.object_vel_idx : self.object_vel_idx + 3]

        object_z_axis = sensors[..., self.object_z_axis_idx : self.object_z_axis_idx + 3]
        object_x_axis = sensors[..., self.object_x_axis_idx : self.object_x_axis_idx + 3]
        object_y_axis = sensors[..., self.object_y_axis_idx : self.object_y_axis_idx + 3]

        # Grasp point proximity vectors
        gripper_to_grasp_left = sensors[..., self.gripper_to_grasp_left_idx : self.gripper_to_grasp_left_idx + 3]
        gripper_to_grasp_right = sensors[..., self.gripper_to_grasp_right_idx : self.gripper_to_grasp_right_idx + 3]

        # Gripper orientation axes
        gripper_x_axis = sensors[..., self.gripper_x_axis_idx : self.gripper_x_axis_idx + 3]
        gripper_y_axis = sensors[..., self.gripper_y_axis_idx : self.gripper_y_axis_idx + 3]

        # Approach site proximity vectors
        torso_to_approach_left = sensors[..., self.torso_to_approach_left_idx : self.torso_to_approach_left_idx + 3]
        torso_to_approach_right = sensors[..., self.torso_to_approach_right_idx : self.torso_to_approach_right_idx + 3]

        fence_violated_x = (body_pos[..., 0] < HARDWARE_FENCE_X[0]) | (body_pos[..., 0] > HARDWARE_FENCE_X[1])
        fence_violated_y = (body_pos[..., 1] < HARDWARE_FENCE_Y[0]) | (body_pos[..., 1] > HARDWARE_FENCE_Y[1])
        spot_fence_reward = -self.config.w_fence * (fence_violated_x | fence_violated_y).any(axis=-1)

        # Check if any state in the rollout has spot fallen
        spot_fallen_reward = -self.config.fall_penalty * (body_height <= self.config.spot_fallen_threshold).any(axis=-1)

        # Compute l2 distance from object pos. to goal.
        goal_reward = -self.config.w_goal * np.linalg.norm(
            object_pos - np.array(self.config.goal_position)[None, None], axis=-1
        ).mean(-1)

        # Reward for uprighting the barrier (z-axis pointing up)
        orientation_alignement = np.minimum(
            np.dot(object_z_axis, Z_AXIS) - 1, self.config.orientation_threshold
        )  # ranging from -2 to 0
        object_orientation_reward = +self.config.w_orientation * np.exp(
            self.config.orientation_sparsity * orientation_alignement
        ).sum(axis=-1)  # ranging from 0 to w_orientation

        # Compute minimum distance from gripper to grasp points (left or right)
        distance_to_left = np.linalg.norm(gripper_to_grasp_left, axis=-1)
        distance_to_right = np.linalg.norm(gripper_to_grasp_right, axis=-1)

        # Stack distances and find minimum across grasp points
        grasp_distances = np.stack([distance_to_left, distance_to_right], axis=-1)
        min_grasp_distance = np.min(grasp_distances, axis=-1)

        # Reward for being close to the nearest grasp point
        gripper_to_grasp_proximity_reward = -self.config.w_gripper_to_grasp_proximity * min_grasp_distance.mean(-1)

        # Gripper orientation reward
        # Barrier frame: X-axis = long dimension (2.191m), Y-axis = thin depth, Z-axis = height
        # Gripper should approach along barrier's long axis: gripper X-axis aligns with barrier X-axis
        # and gripper Y-axis aligns with barrier Z-axis (for upright grasping)
        gripper_x_axis_norm = gripper_x_axis / (np.linalg.norm(gripper_x_axis, axis=-1, keepdims=True) + 1e-8)
        object_x_axis_norm = object_x_axis / (np.linalg.norm(object_x_axis, axis=-1, keepdims=True) + 1e-8)
        x_dot_product = np.abs(np.sum(gripper_x_axis_norm * object_x_axis_norm, axis=-1))
        x_alignment_error = 1.0 - x_dot_product

        gripper_y_axis_norm = gripper_y_axis / (np.linalg.norm(gripper_y_axis, axis=-1, keepdims=True) + 1e-8)
        object_z_axis_norm = object_z_axis / (np.linalg.norm(object_z_axis, axis=-1, keepdims=True) + 1e-8)
        y_dot_product = np.abs(np.sum(gripper_y_axis_norm * object_z_axis_norm, axis=-1))
        y_alignment_error = 1.0 - y_dot_product

        total_orientation_error = x_alignment_error + y_alignment_error
        gripper_orientation_reward = -self.config.w_gripper_orientation * total_orientation_error.mean(axis=-1)

        # Approach site proximity reward - encourage torso to get close to either left or right approach site
        distance_to_left_approach = np.linalg.norm(torso_to_approach_left, axis=-1)
        distance_to_right_approach = np.linalg.norm(torso_to_approach_right, axis=-1)
        approach_distances = np.stack([distance_to_left_approach, distance_to_right_approach], axis=-1)
        min_approach_distance = np.min(approach_distances, axis=-1)
        approach_site_proximity_reward = -self.config.w_approach_site_proximity * min_approach_distance.mean(-1)

        # === Resistance-Based Grasp Detection ===
        # Key insight: If commanded to close but actual position stalls → object is blocking → GRASPING
        # If gripper reaches fully closed (0.0) → no resistance → NOT GRASPING

        gripper_joint_pos = qpos[..., self.gripper_joint_idx]  # Actual position (0.0=closed, -1.54=open)
        gripper_joint_cmd = controls[..., 9]  # ARM_CMD_INDS[6] maps to controls index 9

        # Position error: actual - commanded
        # Positive error means gripper is MORE OPEN than commanded (being blocked by object)
        position_error = gripper_joint_pos - gripper_joint_cmd
        # print(f"position_error shape: {position_error.shape}")
        # Grasp detection logic:
        # 1. Has resistance: gripper can't reach commanded position
        has_resistance = position_error > self.config.resistance_threshold

        # 2. NOT fully closed: if gripper reached 0.0, nothing is blocking it
        is_fully_closed = np.abs(gripper_joint_pos - GRIPPER_CLOSED_POS) < self.config.fully_closed_tolerance
        is_not_empty = ~is_fully_closed

        # 3. Combined: grasping = has resistance AND not empty
        is_grasping = has_resistance & is_not_empty

        # 4. False grasp: gripper is closed but NOT grasping (should be penalized)
        #    This happens when gripper closes in empty space
        is_false_grasp = is_fully_closed & (gripper_joint_cmd < -0.5)  # Commanded to be open but ended up closed

        # Compute squared l2 norm of the object velocity.
        object_linear_velocity_reward = -self.config.w_object_velocity * np.square(
            np.linalg.norm(object_linear_velocity, axis=-1).mean(-1)
        )

        # Compute a velocity penalty to prefer small velocity commands.
        controls_reward = -self.config.w_controls * np.linalg.norm(controls[..., :3], axis=-1).mean(-1)

        # === Grasp-based rewards ===
        # Penalize false grasps (gripper closed but nothing in grasp)
        false_grasp_penalty = -self.config.w_false_grasp_penalty * is_false_grasp.any(axis=-1)

        # Reward for successful grasp (based on resistance strength)
        # Higher position_error = stronger resistance = better grasp
        grasp_quality = np.clip(position_error / 0.5, 0, 1)  # Normalize to [0, 1]
        # print(f"gras_quality shape: {grasp_quality.shape}")
        # print(f"is_grasping shape: {is_grasping.shape}")
        grasp_quality_reward = self.config.w_grasp_quality * (is_grasping * grasp_quality).mean(axis=-1)

        assert spot_fence_reward.shape == (batch_size,)
        assert spot_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert object_orientation_reward.shape == (batch_size,)
        assert gripper_to_grasp_proximity_reward.shape == (batch_size,)
        assert approach_site_proximity_reward.shape == (batch_size,)
        assert object_linear_velocity_reward.shape == (batch_size,)
        assert gripper_orientation_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)
        assert false_grasp_penalty.shape == (batch_size,)
        assert grasp_quality_reward.shape == (batch_size,)

        return (
            # spot_fence_reward
            +spot_fallen_reward
            + goal_reward
            + object_orientation_reward
            + gripper_to_grasp_proximity_reward
            + gripper_orientation_reward
            + approach_site_proximity_reward
            + object_linear_velocity_reward
            + controls_reward
            + false_grasp_penalty  # Penalize closing gripper in empty space
            + grasp_quality_reward  # Reward successful grasp with resistance
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and object."""
        radius = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        # object_pos = np.array([radius * np.cos(theta), radius * np.sin(theta)])
        object_pos = np.array([2.0, 0])
        # Fallen orientation for uprighting task (rotated 90 degrees around y-axis)
        reset_object_pose = np.array([*object_pos, 0.3, np.cos(np.pi / 4), -np.sin(np.pi / 4), 0, 0])

        return np.array(
            [
                # *np.random.randn(2),
                *np.array([0, 0]),
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
        """Check if the crowd barrier is upright, regardless of position."""
        # Get object z-axis sensor data for orientation check
        object_z_axis = data.sensordata[self.object_z_axis_idx : self.object_z_axis_idx + 3]

        # Check orientation tolerance (object should be upright, z-axis aligned with world z-axis)
        orientation_alignment = np.dot(object_z_axis, Z_AXIS)
        orientation_success = orientation_alignment >= (1.0 - self.config.orientation_tolerance)

        return bool(orientation_success)
