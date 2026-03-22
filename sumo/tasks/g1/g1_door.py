# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from judo.utils.fields import np_1d_field
from mujoco import MjData, MjModel

from sumo import MODEL_PATH
from sumo.tasks.g1.g1_base import DEFAULT_JOINT_POSITIONS, STANDING_HEIGHT, G1Base, G1BaseConfig
from sumo.utils.indexing import get_pos_indices, get_sensor_indices

XML_PATH = str(MODEL_PATH / "xml/g1/g1_door.xml")

# Target door angle (in radians) - fully open is -90 degrees (-pi/2, opens away from robot)
TARGET_DOOR_ANGLE = -np.pi / 2  # -90 degrees open

# Success condition tolerances
POSITION_TOLERANCE = 0.3  # XY position tolerance for reaching goal

# Arm usage configuration for door manipulation
USE_LEFT_ARM = False
USE_RIGHT_ARM = True
USE_LEFT_WRIST = False
USE_RIGHT_WRIST = False


@dataclass
class G1DoorConfig(G1BaseConfig):
    """Config for the G1 door opening task."""

    goal_position: np.ndarray = np_1d_field(
        np.array([2.0, 0.0, STANDING_HEIGHT], dtype=np.float64),
        names=["x", "y", "z"],
        mins=[-5.0, -5.0, 0.0],
        maxs=[5.0, 5.0, 2.0],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, None],
    )  # Goal on other side of door
    target_door_angle: float = TARGET_DOOR_ANGLE
    w_goal: float = 30.0  # Weight for reaching goal position (main objective)
    w_pelvis_proximity: float = (
        10  # penalty for pelvis proximity to door (encourage using hands to keep door from pelvis)
    )
    w_hand_proximity: float = 5.0  # Weight for hand proximity to door handle
    w_robot_orientation: float = 50.0  # Weight for keeping robot facing forward (default orientation)
    w_controls: float = 3.0  # Weight for control costs


class G1Door(G1Base):
    config_t = G1DoorConfig
    config: G1DoorConfig

    """Task getting G1 to open a hinged door.

    Uses both arms for manipulation by default (USE_LEFT_ARM=True, USE_RIGHT_ARM=True).
    """

    def __init__(self, model_path: str = XML_PATH) -> None:
        super().__init__(
            model_path=model_path,
            use_left_arm=USE_LEFT_ARM,
            use_right_arm=USE_RIGHT_ARM,
            use_left_wrist=USE_LEFT_WRIST,
            use_right_wrist=USE_RIGHT_WRIST,
        )

        self.body_pose_idx = get_pos_indices(self.model, "floating_base_joint")

        # Get door hinge joint indices
        self.door_hinge_qpos_idx = None
        self.door_hinge_qvel_idx = None
        for i in range(self.model.njnt):
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) == "door_hinge":
                # Find position in qpos and qvel
                self.door_hinge_qpos_idx = self.model.jnt_qposadr[i]
                self.door_hinge_qvel_idx = self.model.jnt_dofadr[i]
                break

        if self.door_hinge_qpos_idx is None:
            raise ValueError("Could not find door_hinge joint in model")

        # Sensor indices
        self.door_hinge_sensor_idx = get_sensor_indices(self.model, "door_hinge_pos")
        self.door_handle_idx = get_sensor_indices(self.model, "trace_door_handle")
        self.left_palm_idx = get_sensor_indices(self.model, "trace_left_palm")
        self.right_palm_idx = get_sensor_indices(self.model, "trace_right_palm")

    @property
    def nu(self) -> int:
        """Number of controls for this task."""
        return super().nu

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for the G1 door opening task."""
        batch_size = states.shape[0]
        config = self.config

        qpos = states[..., : self.model.nq]
        qvel = states[..., self.model.nq : self.model.nq + self.model.nv]

        body_height = qpos[..., self.body_pose_idx[2]]
        body_pos = qpos[..., self.body_pose_idx[0:3]]

        # Get door angle from qpos
        door_angle = qpos[..., self.door_hinge_qpos_idx]

        # Get door angular velocity from qvel
        door_velocity = qvel[..., self.door_hinge_qvel_idx]

        # Get sensor data
        door_handle_pos = sensors[..., self.door_handle_idx]
        right_palm_pos = sensors[..., self.right_palm_idx]

        # Check if any state in the rollout has G1 fallen
        g1_fallen_reward = -config.fall_penalty * (body_height <= config.fall_threshold).any(axis=-1)

        # Goal position reward - MAIN OBJECTIVE: reach the other side of the door
        goal_distance = np.linalg.norm(body_pos - config.goal_position[None, None], axis=-1)
        goal_reward = -config.w_goal * goal_distance.mean(-1)

        # Pelvis proximity to door - encourage getting close to door
        door_position = np.array([2.5, 0.005, 1.015])  # Door center position
        pelvis_to_door_dist = np.linalg.norm(body_pos - door_position[None, None], axis=-1)
        pelvis_proximity_reward = config.w_pelvis_proximity * pelvis_to_door_dist.mean(-1)

        # Hand proximity to door handle - encourage reaching for the handle
        right_hand_to_handle = np.linalg.norm(right_palm_pos - door_handle_pos, axis=-1)
        hand_proximity_reward = -config.w_hand_proximity * right_hand_to_handle.mean(-1)

        # Robot orientation reward - encourage keeping robot facing forward (default orientation)
        # Get robot orientation from quaternion (w, x, y, z format in qpos)
        # The robot starts facing forward (+X direction)
        body_quat = qpos[..., self.body_pose_idx[3:7]]  # (batch, T, 4) - quaternion [w, x, y, z]

        # Calculate forward direction (X-axis) from quaternion
        # For a quaternion [w, x, y, z], the rotated forward vector (originally [1, 0, 0]) is:
        # forward_x = 1 - 2(y^2 + z^2)
        y_q = body_quat[..., 2]
        z_q = body_quat[..., 3]
        forward_x = 1 - 2 * (y_q**2 + z_q**2)

        # Reward when forward_x is positive (facing +X direction, forward)
        # forward_x = 1 means perfectly facing forward (default orientation)
        robot_orientation_reward = config.w_robot_orientation * forward_x.mean(-1)

        # Control cost: penalize velocity commands and arm position deviations
        base_vel = controls[..., :3]
        vel_cost = np.linalg.norm(base_vel, axis=-1).mean(-1)

        if self.use_left_arm or self.use_right_arm:
            # Penalize arm positions as deviation from default
            arm_commands = controls[..., 3:]
            arm_defaults = self.default_command[3:]
            arm_deviation = arm_commands - arm_defaults
            arm_cost = np.linalg.norm(arm_deviation, axis=-1).mean(-1)
            controls_reward = -config.w_controls * (vel_cost + arm_cost)
        else:
            controls_reward = -config.w_controls * vel_cost

        assert g1_fallen_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert pelvis_proximity_reward.shape == (batch_size,)
        assert hand_proximity_reward.shape == (batch_size,)
        assert robot_orientation_reward.shape == (batch_size,)
        assert controls_reward.shape == (batch_size,)

        return (
            g1_fallen_reward
            + goal_reward
            + pelvis_proximity_reward
            + hand_proximity_reward
            + robot_orientation_reward
            + controls_reward
        )

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot and door."""
        # Door starts closed (angle = 0)
        door_angle = 0.0

        # G1 starts in front of the door, slightly offset
        # Door is at x=2.5, so position G1 at x=1.0 (1.5m away)
        robot_x = 0.0
        robot_y = 0.0  # Centered with door

        # G1 reset pose: base position + orientation + joint positions + door angle
        return np.array(
            [
                robot_x,
                robot_y,
                STANDING_HEIGHT,  # Standing height
                1,
                0,
                0,
                0,  # Quaternion (upright orientation, facing door)
                *DEFAULT_JOINT_POSITIONS,  # Joint positions
                door_angle,  # Door hinge angle
            ]
        )

    def success(
        self, model: MjModel, data: MjData, config: G1DoorConfig, metadata: dict[str, Any] | None = None
    ) -> bool:
        """Check if the robot has reached the target XY position."""
        body_pos = data.qpos[..., self.body_pose_idx[0:2]]  # Get XY position only
        goal_pos = np.array(config.goal_position[:2])  # Get XY goal only
        position_check = np.linalg.norm(body_pos - goal_pos, axis=-1) < POSITION_TOLERANCE
        return bool(position_check)

    def failure(
        self, model: MjModel, data: MjData, config: G1DoorConfig, metadata: dict[str, Any] | None = None
    ) -> bool:
        """Check if G1 has fallen."""
        body_height = data.qpos[..., self.body_pose_idx[2]]
        return bool(body_height <= config.fall_threshold)
