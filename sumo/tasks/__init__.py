# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from judo.tasks import get_registered_tasks, register_task

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

register_task("g1_base", G1Base, G1BaseConfig)
register_task("g1_box", G1Box, G1BoxConfig)
register_task("g1_chair_push", G1ChairPush, G1ChairPushConfig)
register_task("g1_door", G1Door, G1DoorConfig)
register_task("g1_table_push", G1TablePush, G1TablePushConfig)

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

register_task("spot_base", SpotBase, SpotBaseConfig)
register_task("spot_box_push", SpotBoxPush, SpotBoxPushConfig)
register_task("spot_chair_push", SpotChairPush, SpotChairPushConfig)
register_task("spot_cone_push", SpotConePush, SpotConePushConfig)
register_task("spot_rack_push", SpotRackPush, SpotRackPushConfig)
register_task("spot_tire_push", SpotTirePush, SpotTirePushConfig)
register_task("spot_box_upright", SpotBoxUpright, SpotBoxUprightConfig)
register_task("spot_chair_upright", SpotChairUpright, SpotChairUprightConfig)
register_task("spot_cone_upright", SpotConeUpright, SpotConeUprightConfig)
register_task("spot_rack_upright", SpotRackUpright, SpotRackUprightConfig)
register_task("spot_tire_upright", SpotTireUpright, SpotTireUprightConfig)
register_task("spot_chair_ramp", SpotChairRamp, SpotChairRampConfig)
register_task("spot_barrier_upright", SpotBarrierUpright, SpotBarrierUprightConfig)
register_task("spot_barrier_drag", SpotBarrierDrag, SpotBarrierDragConfig)
register_task("spot_tire_roll", SpotTireRoll, SpotTireRollConfig)
register_task("spot_tire_stack", SpotTireStack, SpotTireStackConfig)
register_task("spot_tire_rack_drag", SpotTireRackDrag, SpotTireRackDragConfig)
register_task("spot_rugged_box_push", SpotRuggedBoxPush, SpotRuggedBoxPushConfig)

SUMO_TASK_NAMES = G1_TASK_NAMES + SPOT_TASK_NAMES


def get_sumo_registered_tasks() -> dict[str, tuple[type, type]]:
    """Return only the task registrations owned by sumo."""
    registered_tasks = get_registered_tasks()
    return {task_name: registered_tasks[task_name] for task_name in SUMO_TASK_NAMES if task_name in registered_tasks}
