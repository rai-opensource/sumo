# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from judo.tasks import TaskRegistration, get_registered_tasks, register_task
from judo.tasks.spot.spot_constants import SPOT_LOCOMOTION_POLICY_PATH

G1_TASK_NAMES = (
    "g1_base",
    "g1_box",
    "g1_chair_push",
    "g1_door",
    "g1_table_push",
)

from sumo.tasks.g1.g1_base import G1Base, G1BaseConfig
from sumo.tasks.g1.g1_box import G1Box, G1BoxConfig
from sumo.tasks.g1.g1_chair_push import G1ChairPush, G1ChairPushConfig
from sumo.tasks.g1.g1_door import G1Door, G1DoorConfig
from sumo.tasks.g1.g1_table_push import G1TablePush, G1TablePushConfig

register_task("g1_base", G1Base, G1BaseConfig, rollout_backend="mujoco_g1", simulation_backend="mujoco_g1")
register_task("g1_box", G1Box, G1BoxConfig, rollout_backend="mujoco_g1", simulation_backend="mujoco_g1")
register_task(
    "g1_chair_push",
    G1ChairPush,
    G1ChairPushConfig,
    rollout_backend="mujoco_g1",
    simulation_backend="mujoco_g1",
)
register_task("g1_door", G1Door, G1DoorConfig, rollout_backend="mujoco_g1", simulation_backend="mujoco_g1")
register_task(
    "g1_table_push",
    G1TablePush,
    G1TablePushConfig,
    rollout_backend="mujoco_g1",
    simulation_backend="mujoco_g1",
)

SPOT_TASK_NAMES = (
    "spot_base",
    "spot_box_push",
    "spot_chair_push",
    "spot_cone_push",
    "spot_rack_push",
    "spot_tire_push",
    "spot_box_upright",
    "spot_chair_upright",
    "spot_cone_upright",
    "spot_rack_upright",
    "spot_tire_upright",
    "spot_chair_ramp",
    "spot_barrier_upright",
    "spot_barrier_drag",
    "spot_tire_roll",
    "spot_tire_stack",
    "spot_tire_rack_drag",
    "spot_rugged_box_push",
)

# Spot tasks (advanced, using C++ ONNX rollout backend via judo-rai's mujoco_spot)
from sumo.tasks.spot.spot_barrier_drag import SpotBarrierDrag, SpotBarrierDragConfig
from sumo.tasks.spot.spot_barrier_upright import SpotBarrierUpright, SpotBarrierUprightConfig
from sumo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from sumo.tasks.spot.spot_box_push import SpotBoxPush, SpotBoxPushConfig
from sumo.tasks.spot.spot_box_upright import SpotBoxUpright, SpotBoxUprightConfig
from sumo.tasks.spot.spot_chair_push import SpotChairPush, SpotChairPushConfig
from sumo.tasks.spot.spot_chair_ramp import SpotChairRamp, SpotChairRampConfig
from sumo.tasks.spot.spot_chair_upright import SpotChairUpright, SpotChairUprightConfig
from sumo.tasks.spot.spot_cone_push import SpotConePush, SpotConePushConfig
from sumo.tasks.spot.spot_cone_upright import SpotConeUpright, SpotConeUprightConfig
from sumo.tasks.spot.spot_rack_push import SpotRackPush, SpotRackPushConfig
from sumo.tasks.spot.spot_rack_upright import SpotRackUpright, SpotRackUprightConfig
from sumo.tasks.spot.spot_rugged_box_push import SpotRuggedBoxPush, SpotRuggedBoxPushConfig
from sumo.tasks.spot.spot_tire_push import SpotTirePush, SpotTirePushConfig
from sumo.tasks.spot.spot_tire_rack_drag import SpotTireRackDrag, SpotTireRackDragConfig
from sumo.tasks.spot.spot_tire_roll import SpotTireRoll, SpotTireRollConfig
from sumo.tasks.spot.spot_tire_stack import SpotTireStack, SpotTireStackConfig
from sumo.tasks.spot.spot_tire_upright import SpotTireUpright, SpotTireUprightConfig

_SPOT_REGISTRATION_KWARGS = {
    "rollout_backend": "mujoco_hierarchical",
    "simulation_backend": "mujoco_hierarchical",
    "locomotion_policy_path": str(SPOT_LOCOMOTION_POLICY_PATH),
}

register_task("spot_base", SpotBase, SpotBaseConfig, **_SPOT_REGISTRATION_KWARGS)
register_task("spot_box_push", SpotBoxPush, SpotBoxPushConfig, **_SPOT_REGISTRATION_KWARGS)
register_task("spot_chair_push", SpotChairPush, SpotChairPushConfig, **_SPOT_REGISTRATION_KWARGS)
register_task("spot_cone_push", SpotConePush, SpotConePushConfig, **_SPOT_REGISTRATION_KWARGS)
register_task("spot_rack_push", SpotRackPush, SpotRackPushConfig, **_SPOT_REGISTRATION_KWARGS)
register_task("spot_tire_push", SpotTirePush, SpotTirePushConfig, **_SPOT_REGISTRATION_KWARGS)
register_task("spot_box_upright", SpotBoxUpright, SpotBoxUprightConfig, **_SPOT_REGISTRATION_KWARGS)
register_task("spot_chair_upright", SpotChairUpright, SpotChairUprightConfig, **_SPOT_REGISTRATION_KWARGS)
register_task("spot_cone_upright", SpotConeUpright, SpotConeUprightConfig, **_SPOT_REGISTRATION_KWARGS)
register_task("spot_rack_upright", SpotRackUpright, SpotRackUprightConfig, **_SPOT_REGISTRATION_KWARGS)
register_task("spot_tire_upright", SpotTireUpright, SpotTireUprightConfig, **_SPOT_REGISTRATION_KWARGS)
register_task("spot_chair_ramp", SpotChairRamp, SpotChairRampConfig, **_SPOT_REGISTRATION_KWARGS)
register_task("spot_barrier_upright", SpotBarrierUpright, SpotBarrierUprightConfig, **_SPOT_REGISTRATION_KWARGS)
register_task("spot_barrier_drag", SpotBarrierDrag, SpotBarrierDragConfig, **_SPOT_REGISTRATION_KWARGS)
register_task("spot_tire_roll", SpotTireRoll, SpotTireRollConfig, **_SPOT_REGISTRATION_KWARGS)
register_task("spot_tire_stack", SpotTireStack, SpotTireStackConfig, **_SPOT_REGISTRATION_KWARGS)
register_task("spot_tire_rack_drag", SpotTireRackDrag, SpotTireRackDragConfig, **_SPOT_REGISTRATION_KWARGS)
register_task("spot_rugged_box_push", SpotRuggedBoxPush, SpotRuggedBoxPushConfig, **_SPOT_REGISTRATION_KWARGS)

SUMO_TASK_NAMES = G1_TASK_NAMES + SPOT_TASK_NAMES


def get_sumo_registered_tasks() -> dict[str, TaskRegistration]:
    """Return only the task registrations owned by sumo."""
    registered_tasks = get_registered_tasks()
    return {task_name: registered_tasks[task_name] for task_name in SUMO_TASK_NAMES if task_name in registered_tasks}
