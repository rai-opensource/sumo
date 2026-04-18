from dataclasses import fields

import numpy as np
import pytest
from judo.tasks import get_registered_tasks
from mujoco import mj_forward

from sumo.tasks.g1.g1_base import G1Base, G1BaseConfig
from sumo.tasks.g1.g1_box import G1Box, G1BoxConfig
from sumo.tasks.g1.g1_chair_push import G1ChairPush, G1ChairPushConfig
from sumo.tasks.g1.g1_door import G1Door, G1DoorConfig
from sumo.tasks.g1.g1_table_push import G1TablePush, G1TablePushConfig
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
from sumo.tasks.spot.spot_upright import ground_clearance_height

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
    (SpotBoxUpright, SpotBoxUprightConfig),
    (SpotChairUpright, SpotChairUprightConfig),
    (SpotConeUpright, SpotConeUprightConfig),
    (SpotRackUpright, SpotRackUprightConfig),
    (SpotTireUpright, SpotTireUprightConfig),
    (SpotChairRamp, SpotChairRampConfig),
    (SpotBarrierUpright, SpotBarrierUprightConfig),
    (SpotBarrierDrag, SpotBarrierDragConfig),
    (SpotTireRoll, SpotTireRollConfig),
    (SpotTireStack, SpotTireStackConfig),
    (SpotTireRackDrag, SpotTireRackDragConfig),
    (SpotRuggedBoxPush, SpotRuggedBoxPushConfig),
]

SIMPLIFIED_SPOT_PUSH_TASK_CONFIGS = [
    (SpotBoxPush, SpotBoxPushConfig),
    (SpotChairPush, SpotChairPushConfig),
    (SpotConePush, SpotConePushConfig),
    (SpotRackPush, SpotRackPushConfig),
    (SpotTirePush, SpotTirePushConfig),
]

SIMPLIFIED_SPOT_UPRIGHT_TASK_CONFIGS = [
    (SpotBoxUpright, SpotBoxUprightConfig, "box_body", "z"),
    (SpotChairUpright, SpotChairUprightConfig, "yellow_chair", "z"),
    (SpotConeUpright, SpotConeUprightConfig, "traffic_cone", "z"),
    (SpotRackUpright, SpotRackUprightConfig, "tire_rack", "z"),
    (SpotTireUpright, SpotTireUprightConfig, "tire", "horizontal"),
]

REGISTERED_SPOT_TASK_NAMES = [
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


def _get_task_attr(task, *names: str) -> int:
    for name in names:
        if hasattr(task, name):
            return getattr(task, name)
    raise AttributeError(f"{type(task).__name__} has none of {names}")


def _gripper_distance(task, sensors: np.ndarray, object_pos: np.ndarray) -> np.ndarray:
    if hasattr(task, "gripper_pos_idx"):
        gripper_pos_idx = task.gripper_pos_idx
        gripper_pos = sensors[..., gripper_pos_idx : gripper_pos_idx + 3]
        return np.linalg.norm(gripper_pos - object_pos, axis=-1)

    end_effector_to_object_start = _get_task_attr(
        task,
        "end_effector_to_object_idx",
        "end_effector_to_object_start",
    )
    end_effector_to_object = sensors[..., end_effector_to_object_start : end_effector_to_object_start + 3]
    return np.linalg.norm(end_effector_to_object, axis=-1)


@pytest.mark.parametrize(
    "task_cls,config_cls",
    SIMPLIFIED_SPOT_PUSH_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_simplified_spot_push_reward_uses_analysis_terms(task_cls, config_cls):
    task = task_cls()
    states, sensors, controls = _make_spot_rollout_inputs(task)

    qpos = states[..., : task.model.nq]
    object_pose_idx = _get_task_attr(task, "object_pose_idx", "object_pose_start")
    object_vel_idx = _get_task_attr(task, "object_vel_idx", "object_vel_start")
    object_pos = qpos[..., object_pose_idx : object_pose_idx + 3]
    object_linear_velocity = states[..., object_vel_idx : object_vel_idx + 3]

    expected = -task.config.w_goal * np.linalg.norm(
        object_pos - np.asarray(task.config.goal_position)[None, None], axis=-1
    ).mean(-1)
    expected += -task.config.w_gripper_proximity * _gripper_distance(task, sensors, object_pos).mean(-1)
    expected += -task.config.w_object_velocity * np.square(np.linalg.norm(object_linear_velocity, axis=-1).mean(-1))

    np.testing.assert_allclose(task.reward(states, sensors, controls), expected)
    np.testing.assert_allclose(task.reward(states, sensors, controls + 100.0), expected)


@pytest.mark.parametrize(
    "task_cls,config_cls",
    SIMPLIFIED_SPOT_PUSH_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_simplified_spot_push_configs_only_expose_analysis_terms(task_cls, config_cls):
    config_fields = {field.name: field for field in fields(config_cls)}
    assert set(config_fields) == {
        "goal_position",
        "w_goal",
        "w_gripper_proximity",
        "w_object_velocity",
    }
    assert config_fields["w_goal"].type is float
    assert config_fields["w_gripper_proximity"].type is float
    assert config_fields["w_object_velocity"].type is float
    assert config_fields["goal_position"].type is np.ndarray


@pytest.mark.parametrize(
    "task_cls,config_cls,body_name,orientation_kind",
    SIMPLIFIED_SPOT_UPRIGHT_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_simplified_spot_upright_reward_uses_analysis_terms(task_cls, config_cls, body_name, orientation_kind):
    task = task_cls()
    states, sensors, controls = _make_spot_rollout_inputs(task)

    qpos = states[..., : task.model.nq]
    object_pose_idx = _get_task_attr(task, "object_pose_idx", "object_pose_start")
    object_pos = qpos[..., object_pose_idx : object_pose_idx + 3]
    gripper_distance = _gripper_distance(task, sensors, object_pos)

    if orientation_kind == "horizontal":
        axis_idx = _get_task_attr(task, "object_y_axis_idx", "tire_y_axis_idx")
        object_axis = sensors[..., axis_idx : axis_idx + 3]
        orientation_error = np.abs(object_axis[..., 2])
    else:
        axis_idx = task.object_z_axis_idx
        object_axis = sensors[..., axis_idx : axis_idx + 3]
        orientation_error = 1.0 - object_axis[..., 2]

    expected = -task.config.w_orientation * orientation_error.mean(axis=-1)
    expected += -task.config.w_gripper_proximity * gripper_distance.mean(axis=-1)

    np.testing.assert_allclose(task.reward(states, sensors, controls), expected)
    np.testing.assert_allclose(task.reward(states, sensors, controls + 100.0), expected)


@pytest.mark.parametrize(
    "task_cls,config_cls,body_name,orientation_kind",
    SIMPLIFIED_SPOT_UPRIGHT_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_simplified_spot_upright_configs_only_expose_analysis_terms(
    task_cls,
    config_cls,
    body_name,
    orientation_kind,
):
    config_fields = {field.name: field for field in fields(config_cls)}
    assert set(config_fields) == {
        "w_orientation",
        "w_gripper_proximity",
    }
    assert config_fields["w_orientation"].type is float
    assert config_fields["w_gripper_proximity"].type is float


@pytest.mark.parametrize(
    "task_cls,config_cls,body_name,orientation_kind",
    SIMPLIFIED_SPOT_UPRIGHT_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_simplified_spot_upright_reset_randomizes_attitude_without_ground_penetration(
    task_cls,
    config_cls,
    body_name,
    orientation_kind,
):
    task = task_cls()
    object_pose_idx = _get_task_attr(task, "object_pose_idx", "object_pose_start")

    np.random.seed(0)
    reset_pose_a = task.reset_pose
    np.random.seed(1)
    reset_pose_b = task.reset_pose

    quat_a = reset_pose_a[object_pose_idx + 3 : object_pose_idx + 7]
    quat_b = reset_pose_b[object_pose_idx + 3 : object_pose_idx + 7]
    z_a = reset_pose_a[object_pose_idx + 2]

    np.testing.assert_allclose(np.linalg.norm(quat_a), 1.0)
    assert not np.allclose(quat_a, quat_b)
    assert z_a >= ground_clearance_height(task.model, body_name, quat_a)


@pytest.mark.parametrize(
    "task_cls,config_cls,body_name,orientation_kind",
    SIMPLIFIED_SPOT_UPRIGHT_TASK_CONFIGS,
    ids=lambda x: x.__name__ if hasattr(x, "__name__") else str(x),
)
def test_simplified_spot_upright_reset_rejects_already_successful_orientation(
    task_cls,
    config_cls,
    body_name,
    orientation_kind,
    monkeypatch,
):
    task = task_cls()
    object_pose_idx = _get_task_attr(task, "object_pose_idx", "object_pose_start")
    already_successful_quat = np.array([1.0, 0.0, 0.0, 0.0])
    fallen_quat = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
    quat_samples = iter([already_successful_quat, fallen_quat])

    monkeypatch.setattr("sumo.tasks.spot.spot_upright.random_unit_quat", lambda: next(quat_samples))

    reset_pose = task.reset_pose

    np.testing.assert_allclose(reset_pose[object_pose_idx + 3 : object_pose_idx + 7], fallen_quat)
    task.data.qpos[:] = reset_pose
    task.data.qvel[:] = 0.0
    mj_forward(task.model, task.data)
    assert not task.success(task.model, task.data)


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
