# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import mujoco
import numpy as np
from judo.tasks.base import Task, TaskConfig

from sumo import MODEL_PATH
from sumo.utils.indexing import get_pos_indices
from sumo.utils.mujoco import RolloutBackend, SimBackendG1

XML_PATH = str(MODEL_PATH / "xml/g1/g1.xml")

# Default joint positions for G1 (29 joints total)
# Order: left_leg(6) + right_leg(6) + waist(3) + left_arm(7) + right_arm(7)
DEFAULT_JOINT_POSITIONS = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,  # Left leg (6)
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,  # Right leg (6)
        0.0,
        0.0,
        0.0,  # Waist (3)
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,  # Left arm (7)
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,  # Right arm (7)
    ]
)

# Extract default positions for each arm and wrist
# Arm = shoulder (3) + elbow (1) = 4 joints
# Wrist = 3 joints
LEFT_ARM_DEFAULT_POS = DEFAULT_JOINT_POSITIONS[15:19]  # 4 joints (shoulder + elbow)
LEFT_WRIST_DEFAULT_POS = DEFAULT_JOINT_POSITIONS[19:22]  # 3 joints (wrist)
RIGHT_ARM_DEFAULT_POS = DEFAULT_JOINT_POSITIONS[22:26]  # 4 joints (shoulder + elbow)
RIGHT_WRIST_DEFAULT_POS = DEFAULT_JOINT_POSITIONS[26:29]  # 3 joints (wrist)

# Joint names for reference
LEFT_ARM_JOINT_NAMES = ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow"]
LEFT_WRIST_JOINT_NAMES = ["left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw"]
RIGHT_ARM_JOINT_NAMES = ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"]
RIGHT_WRIST_JOINT_NAMES = ["right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]

# Base velocity limits
LIN_VEL_X_RANGE = [-0.25, 0.25]
LIN_VEL_Y_RANGE = [-0.25, 0.25]
ANG_VEL_z_RANGE = [-0.1, 0.1]
STANDING_HEIGHT = 0.76


@dataclass
class G1BaseConfig(TaskConfig):
    """Base config for G1 tasks."""

    fall_penalty: float = 2500.0
    fall_threshold: float = 0.6
    pass


ConfigT = TypeVar("ConfigT", bound=G1BaseConfig)


class G1Base(Task[ConfigT], Generic[ConfigT]):
    """Base class for G1 tasks."""

    config_t: type[G1BaseConfig] = G1BaseConfig  # type: ignore[assignment]
    default_backend = "mujoco_g1"  # Use G1-specific backend

    def _process_spec(self) -> None:
        """No-op for G1 tasks (meshes are local)."""
        pass

    def __init__(
        self,
        model_path: str = XML_PATH,
        use_left_arm: bool = False,
        use_right_arm: bool = False,
        use_left_wrist: bool = False,
        use_right_wrist: bool = False,
    ) -> None:
        super().__init__(model_path)
        self.RolloutBackend = RolloutBackend
        self.SimBackend = SimBackendG1
        self.use_left_arm = use_left_arm
        self.use_right_arm = use_right_arm
        self.use_left_wrist = use_left_wrist
        self.use_right_wrist = use_right_wrist

        self.body_pose_idx = get_pos_indices(self.model, "floating_base_joint")

        # Extract joint control ranges from the model
        self._extract_joint_limits()

        # Set default command based on arm usage
        self._set_default_command()

    def _set_default_command(self) -> None:
        """Set default command based on arm and wrist usage.

        Default command structure:
        - Base velocities: [0, 0, 0] (always present)
        - Left arm (if used): 4 default positions (shoulder + elbow)
        - Left wrist (if used): 3 default positions
        - Right arm (if used): 4 default positions (shoulder + elbow)
        - Right wrist (if used): 3 default positions
        """
        base_vel = np.array([0.0, 0.0, 0.0])
        parts = [base_vel]

        if self.use_left_arm:
            parts.append(LEFT_ARM_DEFAULT_POS)
        if self.use_left_wrist:
            parts.append(LEFT_WRIST_DEFAULT_POS)
        if self.use_right_arm:
            parts.append(RIGHT_ARM_DEFAULT_POS)
        if self.use_right_wrist:
            parts.append(RIGHT_WRIST_DEFAULT_POS)

        self.default_command = np.concatenate(parts)

    def _extract_joint_limits(self) -> None:
        """Extract joint position limits from the MuJoCo model actuators."""
        # Get actuator control ranges
        # Order in model: legs(12) + waist(3) + left_arm(4) + left_wrist(3) + right_arm(4) + right_wrist(3) = 29
        self.joint_ctrl_lower = self.model.actuator_ctrlrange[:, 0]
        self.joint_ctrl_upper = self.model.actuator_ctrlrange[:, 1]

        # Split by joint groups (based on actuator order in XML)
        # Left leg: 0-5, Right leg: 6-11, Waist: 12-14
        # Left arm (shoulder+elbow): 15-18, Left wrist: 19-21
        # Right arm (shoulder+elbow): 22-25, Right wrist: 26-28
        self.left_arm_lower = self.joint_ctrl_lower[15:19]
        self.left_arm_upper = self.joint_ctrl_upper[15:19]
        self.left_wrist_lower = self.joint_ctrl_lower[19:22]
        self.left_wrist_upper = self.joint_ctrl_upper[19:22]
        self.right_arm_lower = self.joint_ctrl_lower[22:26]
        self.right_arm_upper = self.joint_ctrl_upper[22:26]
        self.right_wrist_lower = self.joint_ctrl_lower[26:29]
        self.right_wrist_upper = self.joint_ctrl_upper[26:29]

    @property
    def nu(self) -> int:
        """Number of controls for this task.

        Base velocity: 3 (always)
        Left arm: 4 (shoulder + elbow)
        Left wrist: 3
        Right arm: 4 (shoulder + elbow)
        Right wrist: 3
        """
        count = 3  # Base velocity always present

        if self.use_left_arm:
            count += 4
        if self.use_left_wrist:
            count += 3
        if self.use_right_arm:
            count += 4
        if self.use_right_wrist:
            count += 3

        return count

    @property
    def ctrlrange(self) -> np.ndarray:
        """Control bounds for this task.

        Returns control bounds based on arm/wrist usage:
        - Base velocity: [lin_vel_x, lin_vel_y, ang_vel_z] - always included
        - Left arm (if used): 4 joint position commands (shoulder + elbow)
        - Left wrist (if used): 3 joint position commands
        - Right arm (if used): 4 joint position commands (shoulder + elbow)
        - Right wrist (if used): 3 joint position commands

        Returns:
            Array of shape (nu, 2) where each row is [lower, upper] bound
        """
        # Base velocity limits
        base_lower = np.array([LIN_VEL_X_RANGE[0], LIN_VEL_Y_RANGE[0], ANG_VEL_z_RANGE[0]])
        base_upper = np.array([LIN_VEL_X_RANGE[1], LIN_VEL_Y_RANGE[1], ANG_VEL_z_RANGE[1]])

        # Build control bounds based on what's being used
        lower_parts = [base_lower]
        upper_parts = [base_upper]

        if self.use_left_arm:
            lower_parts.append(self.left_arm_lower)
            upper_parts.append(self.left_arm_upper)
        if self.use_left_wrist:
            lower_parts.append(self.left_wrist_lower)
            upper_parts.append(self.left_wrist_upper)
        if self.use_right_arm:
            lower_parts.append(self.right_arm_lower)
            upper_parts.append(self.right_arm_upper)
        if self.use_right_wrist:
            lower_parts.append(self.right_wrist_lower)
            upper_parts.append(self.right_wrist_upper)

        lower_bound = np.concatenate(lower_parts)
        upper_bound = np.concatenate(upper_parts)

        return np.stack([lower_bound, upper_bound], axis=-1)

    @property
    def actuator_ctrlrange(self) -> np.ndarray:
        """Control bounds in task-control space.

        G1 tasks optimize compact task controls, not the raw 29 actuator commands.
        """
        return self.ctrlrange

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Reward function for G1 tasks with falling penalty.

        Args:
            states: Array of shape (batch_size, horizon+1, nq+nv)
            sensors: Array of shape (batch_size, horizon, nsensordata)
            controls: Array of shape (batch_size, horizon, nu)
            system_metadata: Optional system metadata

        Returns:
            Array of shape (batch_size,) with rewards for each rollout
        """
        batch_size = states.shape[0]
        config = self.config

        # Extract qpos from states
        qpos = states[..., : self.model.nq]

        # Get body height across all timesteps
        body_height = qpos[..., self.body_pose_idx[2]]

        # Check if any state in the rollout has the robot fallen
        # Returns -fall_penalty if fallen at any point, 0 otherwise
        has_fallen = (body_height <= config.fall_threshold).any(axis=-1)
        fallen_reward = -config.fall_penalty * np.asarray(has_fallen)

        assert fallen_reward.shape == (batch_size,), f"Expected shape ({batch_size},), got {fallen_reward.shape}"
        return fallen_reward

    def task_to_sim_ctrl(self, controls: np.ndarray) -> np.ndarray:
        """Map compact controls (..., nu) to 17-dim command expected by C++ rollout.

        Layout of 17-dim command:
        [0:3]   base_vel_cmd (vx, vy, vyaw)
        [3:7]   left_arm_cmd (4 joints: shoulder + elbow)
        [7:10]  left_wrist_cmd (3 joints)
        [10:14] right_arm_cmd (4 joints: shoulder + elbow)
        [14:17] right_wrist_cmd (3 joints)

        If arm/wrist components are not used, their commands are set to 0, which tells
        the C++ backend to use the policy output for those joints.
        """
        controls = np.asarray(controls)
        added_dim = False
        if controls.ndim == 1:
            controls = controls[None]
            added_dim = True

        T = controls.shape[1] if controls.ndim == 3 else 1
        if controls.ndim == 2:
            # assume (..., nu) at sim timestep grid
            controls = controls[:, None, :]
            T = 1

        # Always create 17-dim output
        out = np.zeros((controls.shape[0], controls.shape[1], 17), dtype=controls.dtype)

        # Base velocity (always present)
        out[..., 0:3] = controls[..., 0:3]

        # Map arm and wrist controls based on what's enabled
        # Track current position in the input controls
        idx = 3  # Start after base velocity

        if self.use_left_arm:
            out[..., 3:7] = controls[..., idx : idx + 4]  # Left arm (4 joints)
            idx += 4
        if self.use_left_wrist:
            out[..., 7:10] = controls[..., idx : idx + 3]  # Left wrist (3 joints)
            idx += 3
        if self.use_right_arm:
            out[..., 10:14] = controls[..., idx : idx + 4]  # Right arm (4 joints)
            idx += 4
        if self.use_right_wrist:
            out[..., 14:17] = controls[..., idx : idx + 3]  # Right wrist (3 joints)
            idx += 3

        # Remove dimensions we added
        if T == 1 and out.ndim == 3:
            out = out.squeeze(1)
        if added_dim:
            out = out.squeeze(0)

        return out

    @property
    def reset_pose(self) -> np.ndarray:
        """Reset pose of robot.

        Returns:
            Array of qpos values: [base_pos(3), base_quat(4), joint_pos(29)]
        """
        return np.array(
            [
                0.0,
                0.0,
                STANDING_HEIGHT,  # Base position (x, y, z)
                1.0,
                0.0,
                0.0,
                0.0,  # Base orientation (quaternion: w, x, y, z)
                *DEFAULT_JOINT_POSITIONS,  # All joint positions (29)
            ]
        )

    def reset(self) -> None:
        """Reset the robot to its default standing pose."""
        self.data.qpos = self.reset_pose
        self.data.qvel = np.zeros_like(self.data.qvel)
        mujoco.mj_forward(self.model, self.data)
