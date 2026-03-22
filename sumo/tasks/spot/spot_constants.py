# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np

# Import shared constants from judo-rai
from judo.tasks.spot.spot_constants import (  # noqa: F401
    ARM_JOINT_NAMES,
    ARM_STOWED_POS,
    ARM_UNSTOWED_POS,
    GRIPPER_CLOSED_POS,
    GRIPPER_OPEN_POS,
    LEG_JOINT_NAMES_BOSDYN,
    LEGS_STANDING_POS,
    LEGS_STANDING_POS_RL,
    STANDING_HEIGHT,
    TIRE_HALF_WIDTH,
    TIRE_RADIUS,
    Z_AXIS,
)

LEG_JOINT_NAMES_ISAAC = [
    "fl_hx",
    "fr_hx",
    "hl_hx",
    "hr_hx",
    "fl_hy",
    "fr_hy",
    "hl_hy",
    "hr_hy",
    "fl_kn",
    "fr_kn",
    "hl_kn",
    "hr_kn",
]

JOINT_NAMES_BOSDYN = LEG_JOINT_NAMES_BOSDYN + ARM_JOINT_NAMES

JOINT_NAMES_ISAAC = [
    "arm_sh0",
    "fl_hx",
    "fr_hx",
    "hl_hx",
    "hr_hx",
    "arm_sh1",
    "fl_hy",
    "fr_hy",
    "hl_hy",
    "hr_hy",
    "arm_el0",
    "fl_kn",
    "fr_kn",
    "hl_kn",
    "hr_kn",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
    "arm_f1x",
]

LEG_NAMES = [
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
]

FEET_NAMES = [f"{leg_name}_foot" for leg_name in LEG_NAMES]

FL_JOINT_NAMES = ["fl_hx", "fl_hy", "fl_kn"]
FR_JOINT_NAMES = ["fr_hx", "fr_hy", "fr_kn"]
HL_JOINT_NAMES = ["hl_hx", "hl_hy", "hl_kn"]
HR_JOINT_NAMES = ["hr_hx", "hr_hy", "hr_kn"]
LEG_JOINT_NAMES_DICT = {
    "fl": FL_JOINT_NAMES,
    "fr": FR_JOINT_NAMES,
    "hl": HL_JOINT_NAMES,
    "hr": HR_JOINT_NAMES,
}


def get_joint_names_bosdyn(has_arm: bool) -> list:
    """Return the list of Spot joint names in BD API order.

    Args:
        has_arm (bool): If true, include the arm joints.
    """
    if has_arm:
        return JOINT_NAMES_BOSDYN
    return LEG_JOINT_NAMES_BOSDYN


def get_joint_names_isaac(has_arm: bool) -> list:
    """Return the list of Spot joint names in Isaac order.

    Args:
        has_arm (bool): If true, include the arm joints.
    """
    if has_arm:
        return JOINT_NAMES_ISAAC
    return LEG_JOINT_NAMES_ISAAC


### Camera-related names
CAMERA_IMAGE_SOURCES = [
    "frontleft_fisheye_image",
    "frontright_fisheye_image",
    "left_fisheye_image",
    "right_fisheye_image",
    "back_fisheye_image",
    "hand_color_image",
]
DEPTH_IMAGE_SOURCES = [
    "frontleft_depth",
    "frontright_depth",
    "left_depth",
    "right_depth",
    "back_depth",
    "hand_depth",
]
DEPTH_REGISTERED_IMAGE_SOURCES = [
    "frontleft_depth_in_visual_frame",
    "frontright_depth_in_visual_frame",
    "right_depth_in_visual_frame",
    "left_depth_in_visual_frame",
    "back_depth_in_visual_frame",
    "hand_depth_in_hand_color_frame",
]

IMAGE_TYPES = {"visual", "depth", "depth_registered"}


### Enums and DataClasses
class DOF(IntEnum):
    """Link index and order"""

    # FL_HX = spot_constants_pb2.JOINT_INDEX_FL_HX
    # FL_HY = spot_constants_pb2.JOINT_INDEX_FL_HY
    # FL_KN = spot_constants_pb2.JOINT_INDEX_FL_KN
    # FR_HX = spot_constants_pb2.JOINT_INDEX_FR_HX
    # FR_HY = spot_constants_pb2.JOINT_INDEX_FR_HY
    # FR_KN = spot_constants_pb2.JOINT_INDEX_FR_KN
    # HL_HX = spot_constants_pb2.JOINT_INDEX_HL_HX
    # HL_HY = spot_constants_pb2.JOINT_INDEX_HL_HY
    # HL_KN = spot_constants_pb2.JOINT_INDEX_HL_KN
    # HR_HX = spot_constants_pb2.JOINT_INDEX_HR_HX
    # HR_HY = spot_constants_pb2.JOINT_INDEX_HR_HY
    # HR_KN = spot_constants_pb2.JOINT_INDEX_HR_KN
    # # Arm
    # A0_SH0 = spot_constants_pb2.JOINT_INDEX_A0_SH0
    # A0_SH1 = spot_constants_pb2.JOINT_INDEX_A0_SH1
    # A0_EL0 = spot_constants_pb2.JOINT_INDEX_A0_EL0
    # A0_EL1 = spot_constants_pb2.JOINT_INDEX_A0_EL1
    # A0_WR0 = spot_constants_pb2.JOINT_INDEX_A0_WR0
    # A0_WR1 = spot_constants_pb2.JOINT_INDEX_A0_WR1
    # # Hand
    # A0_F1X = spot_constants_pb2.JOINT_INDEX_A0_F1X

    # DOF count for strictly the legs.
    N_DOF_LEGS = 12
    # DOF count for all DOF on robot (arms and legs).
    N_DOF = 19


class LEGS(IntEnum):
    """Leg links index and order"""

    # FL = spot_constants_pb2.LEG_INDEX_FL
    # FR = spot_constants_pb2.LEG_INDEX_FR
    # HL = spot_constants_pb2.LEG_INDEX_HL
    # HR = spot_constants_pb2.LEG_INDEX_HR

    N_LEGS = 4


class LegDofOrder(IntEnum):
    """Legs DoF order"""

    # HX = spot_constants_pb2.HX
    # HY = spot_constants_pb2.HY
    # KN = spot_constants_pb2.KN

    # The number of leg dof
    N_LEG_DOF = 3


@dataclass
class POLICY_MODE:
    """Dataclass containing the two supported policy modes"""

    boston_dynamics: str = "boston_dynamics"
    rl_locomotion: str = "rl_locomotion"


### Gains
# Bosdyn
LEG_K_Q_P_BOSDYN = [624, 936, 286.0] * LEGS.N_LEGS
LEG_K_QD_P_BOSDYN = [5.20, 5.20, 2.04] * LEGS.N_LEGS
ARM_K_Q_P_BOSDYN = [1020, 255, 204, 102, 102, 102, 16.0]
ARM_K_QD_P_BOSDYN = [10.2, 15.3, 10.2, 2.04, 2.04, 2.04, 0.32]

# RL locomotion policy during training
LEG_K_Q_P_RL = [60, 60, 60.0] * LEGS.N_LEGS
LEG_K_QD_P_RL = [1.5, 1.5, 1.5] * LEGS.N_LEGS
ARM_K_Q_P_RL = [120, 120, 120, 100, 100, 100, 16.0]
ARM_K_QD_P_RL = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.32]
K_Q_P_RL = LEG_K_Q_P_RL + ARM_K_Q_P_RL
K_QD_P_RL = LEG_K_QD_P_RL + ARM_K_QD_P_RL

# RL locomotion policy on hardware
LEG_K_Q_P_HW = [75, 75, 75.0] * LEGS.N_LEGS
LEG_K_QD_P_HW = LEG_K_QD_P_RL
ARM_K_Q_P_HW = ARM_K_Q_P_RL
ARM_K_QD_P_HW = ARM_K_QD_P_RL

### Initial Configurations
# Standard joint positions
LEGS_SITTING_POS = np.array(
    [
        0.48,
        1.26,
        -2.7929,
        -0.48,
        1.26,
        -2.7929,
        0.48,
        1.26,
        -2.7929,
        -0.48,
        1.26,
        -2.7929,
    ]
)

ARM_MIDDLE_POS = np.array(
    [
        0,
        -2.8,
        2.8,
        1.56,
        0,
        -1.56,
        GRIPPER_CLOSED_POS,
    ]
)

SITTING_STOWED_POS = np.concatenate((LEGS_SITTING_POS, ARM_STOWED_POS))

SITTING_UNSTOWED_POS = np.concatenate((LEGS_SITTING_POS, ARM_UNSTOWED_POS))

STANDING_STOWED_POS = np.concatenate((LEGS_STANDING_POS, ARM_STOWED_POS))

STANDING_UNSTOWED_POS = np.concatenate((LEGS_STANDING_POS, ARM_UNSTOWED_POS))

STANDING_POS_RL = np.concatenate((LEGS_STANDING_POS_RL, ARM_UNSTOWED_POS[:-1], np.array([GRIPPER_OPEN_POS])))

SITTING_HEIGHT = 0.1
STANDING_HEIGHT_CMD = 0.55

### Limits
# RL locomotion policy limits
LEG_TORQUE_LIMITS_RL = [45.0, 45.0, 60.0] * LEGS.N_LEGS
ARM_TORQUE_LIMITS_RL = [90.9, 181.8, 90.9, 30.3, 30.3, 30.3, 15.32]
TORQUE_LIMITS_RL = LEG_TORQUE_LIMITS_RL + ARM_TORQUE_LIMITS_RL

# position limits
LEG_LOWER_JOINT_LIMITS = np.array([-0.785398, -0.898845, -2.7929] * LEGS.N_LEGS)
LEG_UPPER_JOINT_LIMITS = np.array([0.785398, 2.29511, -0.254801] * LEGS.N_LEGS)
ARM_LOWER_JOINT_LIMITS = np.array([-2.618, -3.1416, 0, -2.7925, -1.8326, -2.8798, -1.57])
ARM_UPPER_JOINT_LIMITS = np.array([3.1416, 0.5236, 3.1416, 2.7925, 1.8326, 2.8798, 0])
LOWER_JOINT_LIMITS = np.concatenate((LEG_LOWER_JOINT_LIMITS, ARM_LOWER_JOINT_LIMITS))
UPPER_JOINT_LIMITS = np.concatenate((LEG_UPPER_JOINT_LIMITS, ARM_UPPER_JOINT_LIMITS))

# velocity limits
VELOCITY_TASK_SAFETY_LIMIT: float = 30.0  # Limit when task execution should be stopped
VELOCITY_HW_SAFETY_LIMIT: float = 40.0  # Limit when Spot will shut down

# Command lengths
# velocity_cmd: 3, arm_joint_cmd: 7, leg_joint_cmd: 12, torso_rph: 3 --> total = 25
RL_LOCOMOTION_COMMAND_LENGTH = 25
RL_LOCOMOTION_ACTION_LENGTH = 12

SECOND_TO_NANOSECOND = 1_000_000_000


### Target info
# For Robot IPs from, see:
# https://www.notion.so/theaiinstitute/Spot-Passwords-and-Network-Info-09fb3beacb6440d1bcc3677ae532307c

P2P_IP_ADDRESS = "192.168.50.3"


@dataclass
class TargetInfo:
    """Information for targets to run policies on"""

    mocap_id: int
    eth_ip: str = ""
    p2p_ip: str = P2P_IP_ADDRESS

    def __post_init__(self) -> None:
        if self.eth_ip == "":
            self.eth_ip = f"10.0.0.{self.mocap_id}"


class ValidTargets(Enum):
    """Targets to run policies on"""

    # Mujoco Mocap ID for needed for rendering. IP address is placeholder.
    mujoco = TargetInfo(mocap_id=100)
    quinn = TargetInfo(mocap_id=101)
    fausto = TargetInfo(mocap_id=102)
    seashell = TargetInfo(mocap_id=103)
    gosu = TargetInfo(mocap_id=104)
    zephyr = TargetInfo(mocap_id=105)
    lionel = TargetInfo(mocap_id=106)
    bard = TargetInfo(mocap_id=107)
    antora = TargetInfo(mocap_id=108)

    @classmethod
    def is_valid_target(cls, target: str) -> bool:
        """Check if string is valid target"""
        return target in cls.__members__

    @classmethod
    def is_valid_mocap_id(cls, mocap_id: int) -> bool:
        """Check if string is valid target"""
        return mocap_id in [cls[target].value.mocap_id for target in cls.__members__]

    @classmethod
    def helptext(cls) -> str:
        """Help text for ValidTarget command line argument."""
        return f"Robot target. Choose from:\n [{', '.join([x.name for x in cls])}]"


### Isaac <-> Mujoco conversion
ISAAC_TO_MUJOCO_INDICES_12 = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
ISAAC_TO_MUJOCO_INDICES_19 = [1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 0, 5, 10, 15, 16, 17, 18]
MUJOCO_TO_ISAAC_INDICES_12 = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
MUJOCO_TO_ISAAC_INDICES_19 = [12, 0, 3, 6, 9, 13, 1, 4, 7, 10, 14, 2, 5, 8, 11, 15, 16, 17, 18]


def isaac_to_mujoco(var_isaac: np.ndarray) -> np.ndarray:
    """Reorders an array from Isaac Gym to MuJoCo format.

    Args:
        var_isaac (np.ndarray): Input array in Isaac Gym format.

    Returns:
        np.ndarray: Array reordered for MuJoCo.
    """
    if len(var_isaac) == 12:
        isaac_to_mujoco_list = ISAAC_TO_MUJOCO_INDICES_12
    elif len(var_isaac) == 19:
        isaac_to_mujoco_list = ISAAC_TO_MUJOCO_INDICES_19
    else:
        raise ValueError(
            f"The vector must of size 12 (legs only) or 19 (arm + legs), size of given vector {var_isaac.shape}"
        )
    var_mujoco = var_isaac[isaac_to_mujoco_list]
    return var_mujoco


def mujoco_to_isaac(var_mujoco: np.ndarray) -> np.ndarray:
    """Reorders an array from MuJoCo to Isaac Gym format.

    Args:
        var_mujoco (np.ndarray): Input array in MuJoCo format.

    Returns:
        np.ndarray: Array reordered for Isaac Gym.
    """
    if len(var_mujoco) == 12:
        mujoco_to_isaac_list = MUJOCO_TO_ISAAC_INDICES_12
    elif len(var_mujoco) == 19:
        mujoco_to_isaac_list = MUJOCO_TO_ISAAC_INDICES_19
    else:
        raise ValueError(
            f"The vector must of size 12 (legs only) or 19 (arm + legs), size of given vector {var_mujoco.shape}"
        )
    var_isaac = var_mujoco[mujoco_to_isaac_list]
    return var_isaac


### Spot and object states
def _slice_union(*slices: slice) -> slice:
    """Create a union of multiple slices, ensuring they are contiguous."""
    # Sort slices based on their start values
    sorted_slices = sorted(slices, key=lambda s: s.start)

    start = sorted_slices[0].start
    stop = sorted_slices[-1].stop

    # Check for contiguity
    for i in range(len(sorted_slices) - 1):
        if sorted_slices[i].stop != sorted_slices[i + 1].start:
            raise ValueError("Slices are not contiguous")

    return slice(start, stop)


def _non_contiguous_slices_to_indices(*slices: slice) -> list[int]:
    """Create a union of multiple slices that are not contiguous."""
    # Sort slices based on their start values
    sorted_slices = sorted(slices, key=lambda s: s.start)
    indices: list[int] = []
    for s in sorted_slices:
        indices.extend(range(s.start, s.stop))
    return indices


@dataclass(frozen=True)
class SpotObjectStateArray:
    """State array indexing helper for spot object MuJoCo state."""

    # TODO: make SceneStateArray - a dynamic indexing helper based on the loaded model

    # Position slices
    base_pos = slice(0, 3)
    base_quat = slice(3, 7)
    fl_leg = slice(7, 10)
    fr_leg = slice(10, 13)
    hl_leg = slice(13, 16)
    hr_leg = slice(16, 19)
    arm = slice(19, 26)
    object_pos = slice(26, 29)
    object_quat = slice(29, 33)

    # Velocity slices
    base_lin_vel = slice(33, 36)
    base_ang_vel = slice(36, 39)
    fl_leg_vel = slice(39, 42)
    fr_leg_vel = slice(42, 45)
    hl_leg_vel = slice(45, 48)
    hr_leg_vel = slice(48, 51)
    arm_vel = slice(51, 58)
    object_lin_vel = slice(58, 61)
    object_ang_vel = slice(61, 64)

    ## Unions
    # Positions
    base_pose = _slice_union(base_pos, base_quat)
    legs = _slice_union(fl_leg, fr_leg, hl_leg, hr_leg)
    joint_pos = _slice_union(legs, arm)
    object_pose = _slice_union(object_pos, object_quat)
    qpos = _slice_union(base_pose, joint_pos, object_pose)
    # Velocities
    base_vel = _slice_union(base_lin_vel, base_ang_vel)
    legs_vel = _slice_union(fl_leg_vel, fr_leg_vel, hl_leg_vel, hr_leg_vel)
    joint_vel = _slice_union(legs_vel, arm_vel)
    object_vel = _slice_union(object_lin_vel, object_ang_vel)
    qvel = _slice_union(base_vel, joint_vel, object_vel)
    # Full States
    spot_state = _non_contiguous_slices_to_indices(base_pose, joint_pos, base_vel, joint_vel)
    object_state = _non_contiguous_slices_to_indices(object_pose, object_vel)

    size = 64


# TODO(tong): Is this even used anywhere? Why is this still around?
@dataclass(frozen=True)
class SpotStateArray:
    """State array indexing helper for spot MuJoCo state."""

    # TODO: make SceneStateArray - a dynamic indexing helper based on the loaded model

    # Position slices
    base_pos = slice(0, 3)
    base_quat = slice(3, 7)
    fl_leg = slice(7, 10)
    fr_leg = slice(10, 13)
    hl_leg = slice(13, 16)
    hr_leg = slice(16, 19)
    arm = slice(19, 26)

    # Velocity slices
    base_lin_vel = slice(26, 29)
    base_ang_vel = slice(29, 32)
    fl_leg_vel = slice(32, 35)
    fr_leg_vel = slice(35, 38)
    hl_leg_vel = slice(38, 41)
    hr_leg_vel = slice(41, 44)
    arm_vel = slice(44, 51)

    ## Unions
    # Positions
    base_pose = _slice_union(base_pos, base_quat)
    legs = _slice_union(fl_leg, fr_leg, hl_leg, hr_leg)
    joint_pos = _slice_union(legs, arm)
    qpos = _slice_union(base_pose, joint_pos)
    # Velocities
    base_vel = _slice_union(base_lin_vel, base_ang_vel)
    legs_vel = _slice_union(fl_leg_vel, fr_leg_vel, hl_leg_vel, hr_leg_vel)
    joint_vel = _slice_union(legs_vel, arm_vel)
    qvel = _slice_union(base_vel, joint_vel)
    # Full States
    spot_state = _slice_union(qpos, qvel)

    # Object and Goal None values for type checking
    object_pos = None
    object_quat = None
    object_vel = None

    size = 51


# Foxglove joint states update schema
joint_state_json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "array", "items": {"type": "string"}},
        "position": {"type": "array", "items": {"type": "number"}},
        "velocity": {"type": "array", "items": {"type": "number"}},
        "effort": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["name", "position", "velocity", "effort"],
}
