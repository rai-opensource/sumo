import numpy as np
import pytest
from judo.tasks import get_registered_tasks

from sumo.tasks.g1.g1_base import G1Base, G1BaseConfig
from sumo.tasks.g1.g1_box import G1Box, G1BoxConfig
from sumo.tasks.g1.g1_chair_push import G1ChairPush, G1ChairPushConfig
from sumo.tasks.g1.g1_door import G1Door, G1DoorConfig
from sumo.tasks.g1.g1_table_push import G1TablePush, G1TablePushConfig
from sumo.tasks.spot.spot_barrier_drag import SpotBarrierDrag, SpotBarrierDragConfig
from sumo.tasks.spot.spot_barrier_upright import SpotBarrierUpright, SpotBarrierUprightConfig
from sumo.tasks.spot.spot_base import SpotBase, SpotBaseConfig
from sumo.tasks.spot.spot_box_push import SpotBoxPush, SpotBoxPushConfig
from sumo.tasks.spot.spot_chair_push import SpotChairPush, SpotChairPushConfig
from sumo.tasks.spot.spot_chair_ramp import SpotChairRamp, SpotChairRampConfig
from sumo.tasks.spot.spot_cone_push import SpotConePush, SpotConePushConfig
from sumo.tasks.spot.spot_cone_upright import SpotConeUpright, SpotConeUprightConfig
from sumo.tasks.spot.spot_rack_push import SpotRackPush, SpotRackPushConfig
from sumo.tasks.spot.spot_rugged_box_push import SpotRuggedBoxPush, SpotRuggedBoxPushConfig
from sumo.tasks.spot.spot_tire_push import SpotTirePush, SpotTirePushConfig
from sumo.tasks.spot.spot_tire_rack_drag import SpotTireRackDrag, SpotTireRackDragConfig
from sumo.tasks.spot.spot_tire_roll import SpotTireRoll, SpotTireRollConfig
from sumo.tasks.spot.spot_tire_stack import SpotTireStack, SpotTireStackConfig
from sumo.tasks.spot.spot_tire_upright import SpotTireUpright, SpotTireUprightConfig

G1_TASK_CONFIGS = [
    (G1Base, G1BaseConfig, 3),
    (G1Box, G1BoxConfig, 3),
    (G1ChairPush, G1ChairPushConfig, 3),
    (G1Door, G1DoorConfig, 7),
    (G1TablePush, G1TablePushConfig, 11),
]

SPOT_TASK_CONFIGS = [
    (SpotBase, SpotBaseConfig),
    (SpotBoxPush, SpotBoxPushConfig),
    (SpotChairPush, SpotChairPushConfig),
    (SpotConePush, SpotConePushConfig),
    (SpotRackPush, SpotRackPushConfig),
    (SpotTirePush, SpotTirePushConfig),
    (SpotConeUpright, SpotConeUprightConfig),
    (SpotChairRamp, SpotChairRampConfig),
    (SpotBarrierUpright, SpotBarrierUprightConfig),
    (SpotBarrierDrag, SpotBarrierDragConfig),
    (SpotTireRoll, SpotTireRollConfig),
    (SpotTireUpright, SpotTireUprightConfig),
    (SpotTireStack, SpotTireStackConfig),
    (SpotTireRackDrag, SpotTireRackDragConfig),
    (SpotRuggedBoxPush, SpotRuggedBoxPushConfig),
]

REGISTERED_SPOT_TASK_NAMES = [
    "spot_base",
    "spot_box_push",
    "spot_chair_push",
    "spot_cone_push",
    "spot_rack_push",
    "spot_tire_push",
    "spot_cone_upright",
    "spot_chair_ramp",
    "spot_barrier_upright",
    "spot_barrier_drag",
    "spot_tire_roll",
    "spot_tire_upright",
    "spot_tire_stack",
    "spot_tire_rack_drag",
    "spot_rugged_box_push",
]


def _make_g1_rollout_inputs(task, seed: int = 0):
    rng = np.random.default_rng(seed)
    batch_size = 4
    horizon = 10
    states = rng.standard_normal((batch_size, horizon + 1, task.model.nq + task.model.nv))
    sensors = rng.standard_normal((batch_size, horizon, task.model.nsensordata))
    controls = rng.standard_normal((batch_size, horizon, task.nu))
    return states, sensors, controls


def _make_spot_rollout_inputs(task, seed: int = 0):
    rng = np.random.default_rng(seed)
    batch_size = 4
    horizon = 10
    states = rng.standard_normal((batch_size, horizon, task.model.nq + task.model.nv))
    sensors = rng.standard_normal((batch_size, horizon, task.model.nsensordata))
    controls = rng.standard_normal((batch_size, horizon, task.nu))
    return states, sensors, controls


@pytest.mark.parametrize(
    "task_cls,config_cls,expected_nu",
    G1_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_g1_task_instantiation(task_cls, config_cls, expected_nu):
    task = task_cls()
    assert task.model is not None
    assert task.nu == expected_nu
    assert task.config_t is config_cls
    assert isinstance(task.config, config_cls)


@pytest.mark.parametrize(
    "task_cls,config_cls,expected_nu",
    G1_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_g1_task_reset(task_cls, config_cls, expected_nu):
    task = task_cls()
    task.reset()
    np.testing.assert_array_equal(task.data.qpos, task.reset_pose)
    np.testing.assert_array_equal(task.data.qvel, np.zeros(task.model.nv))


@pytest.mark.parametrize(
    "task_cls,config_cls,expected_nu",
    G1_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_g1_task_to_sim_ctrl(task_cls, config_cls, expected_nu):
    task = task_cls()
    controls = np.zeros(expected_nu)
    result = task.task_to_sim_ctrl(controls)
    assert result.shape == (17,)


@pytest.mark.parametrize(
    "task_cls,config_cls,expected_nu",
    G1_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_g1_task_reward_shape(task_cls, config_cls, expected_nu):
    task = task_cls()
    states, sensors, controls = _make_g1_rollout_inputs(task)
    reward = task.reward(states, sensors, controls)
    assert reward.shape == (states.shape[0],)


@pytest.mark.parametrize(
    "task_cls,config_cls,expected_nu",
    G1_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_g1_task_pre_rollout(task_cls, config_cls, expected_nu):
    task = task_cls()
    task.reset()
    curr_state = np.concatenate([task.data.qpos, task.data.qvel])
    task.pre_rollout(curr_state)


@pytest.mark.parametrize(
    "task_cls,config_cls,expected_nu",
    G1_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_g1_task_ctrlrange(task_cls, config_cls, expected_nu):
    task = task_cls()
    ctrlrange = task.ctrlrange
    assert ctrlrange.shape == (expected_nu, 2)
    assert np.all(ctrlrange[:, 0] <= ctrlrange[:, 1])


@pytest.mark.parametrize(
    "task_cls,config_cls,expected_nu",
    G1_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_g1_task_actuator_ctrlrange(task_cls, config_cls, expected_nu):
    task = task_cls()
    ctrlrange = task.actuator_ctrlrange
    assert ctrlrange.shape == (expected_nu, 2)
    assert np.all(ctrlrange[:, 0] <= ctrlrange[:, 1])


@pytest.mark.parametrize(
    "task_cls,config_cls",
    SPOT_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_spot_task_instantiation(task_cls, config_cls):
    task = task_cls()
    assert task.model is not None
    assert task.nu > 0


@pytest.mark.parametrize(
    "task_cls,config_cls",
    SPOT_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_spot_task_reset(task_cls, config_cls):
    task = task_cls()
    task.reset()
    assert len(task.data.qpos) == task.model.nq
    np.testing.assert_array_equal(task.data.qvel, np.zeros(task.model.nv))


@pytest.mark.parametrize(
    "task_cls,config_cls",
    SPOT_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_spot_task_to_sim_ctrl(task_cls, config_cls):
    task = task_cls()
    result = task.task_to_sim_ctrl(np.zeros(task.nu))
    assert result.shape[-1] == 25


@pytest.mark.parametrize(
    "task_cls,config_cls",
    SPOT_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_spot_task_reward_shape(task_cls, config_cls):
    task = task_cls()
    states, sensors, controls = _make_spot_rollout_inputs(task)
    reward = task.reward(states, sensors, controls)
    assert reward.shape == (states.shape[0],)


@pytest.mark.parametrize(
    "task_cls,config_cls",
    SPOT_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_spot_task_ctrlrange(task_cls, config_cls):
    task = task_cls()
    ctrlrange = task.actuator_ctrlrange
    assert ctrlrange.shape == (task.nu, 2)
    assert np.all(ctrlrange[:, 0] <= ctrlrange[:, 1])


@pytest.mark.parametrize("task_name", REGISTERED_SPOT_TASK_NAMES)
def test_registered_spot_task_instantiation(task_name):
    task_cls, config_cls = get_registered_tasks()[task_name]
    task = task_cls()
    assert task.model is not None
    assert isinstance(task.config, config_cls)


@pytest.mark.g1_extensions
def test_g1_controller_uses_g1_backend():
    from sumo.controller import make_controller
    from sumo.utils.mujoco import G1RolloutBackend

    controller = make_controller("g1_box", "cem")

    assert isinstance(controller.rollout_backend, G1RolloutBackend)
    assert controller.action_normalizer.dim == controller.task.nu
