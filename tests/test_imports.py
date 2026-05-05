import importlib
import os
from pathlib import Path

import pytest

SPOT_TASK_MODULES = [
    "sumo.tasks.spot.spot_base",
    "sumo.tasks.spot.spot_box_push",
    "sumo.tasks.spot.spot_chair_push",
    "sumo.tasks.spot.spot_cone_push",
    "sumo.tasks.spot.spot_rack_push",
    "sumo.tasks.spot.spot_tire_push",
    "sumo.tasks.spot.spot_box_upright",
    "sumo.tasks.spot.spot_chair_upright",
    "sumo.tasks.spot.spot_cone_upright",
    "sumo.tasks.spot.spot_rack_upright",
    "sumo.tasks.spot.spot_chair_ramp",
    "sumo.tasks.spot.spot_barrier_upright",
    "sumo.tasks.spot.spot_barrier_drag",
    "sumo.tasks.spot.spot_tire_roll",
    "sumo.tasks.spot.spot_tire_upright",
    "sumo.tasks.spot.spot_tire_stack",
    "sumo.tasks.spot.spot_tire_rack_drag",
    "sumo.tasks.spot.spot_rugged_box_push",
]

G1_TASK_NAMES = [
    "g1_base",
    "g1_box",
    "g1_chair_push",
    "g1_door",
    "g1_table_push",
]

SPOT_TASK_NAMES = [
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

G1_IMPORTS = {
    "sumo.tasks.g1.g1_base": ["G1Base", "G1BaseConfig"],
    "sumo.tasks.g1.g1_box": ["G1Box", "G1BoxConfig"],
    "sumo.tasks.g1.g1_chair_push": ["G1ChairPush", "G1ChairPushConfig"],
    "sumo.tasks.g1.g1_door": ["G1Door", "G1DoorConfig"],
    "sumo.tasks.g1.g1_table_push": ["G1TablePush", "G1TablePushConfig"],
}


def test_public_judo_resolution():
    import judo

    judo_path = Path(judo.__file__).resolve()
    assert "judo-private" not in str(judo_path)


def test_sumo_package():
    import sumo

    assert hasattr(sumo, "PACKAGE_ROOT")
    assert hasattr(sumo, "MODEL_PATH")
    assert sumo.MODEL_PATH.exists()


def test_sumo_utils():
    from sumo.utils import indexing, mujoco

    assert hasattr(indexing, "get_pos_indices")
    assert hasattr(indexing, "get_sensor_indices")
    assert hasattr(indexing, "get_vel_indices")
    assert hasattr(mujoco, "G1RolloutBackend")
    assert hasattr(mujoco, "make_model_data_pairs")


def test_sumo_tasks_import():
    assert importlib.import_module("sumo.tasks") is not None

    for module_name, attrs in G1_IMPORTS.items():
        module = importlib.import_module(module_name)
        for attr in attrs:
            assert hasattr(module, attr)


@pytest.mark.parametrize("module_name", SPOT_TASK_MODULES)
def test_spot_task_module_imports(module_name):
    assert importlib.import_module(module_name) is not None


def test_tasks_registered():
    from judo.tasks import get_registered_tasks

    importlib.import_module("sumo.tasks")
    tasks = get_registered_tasks()
    for name in G1_TASK_NAMES + SPOT_TASK_NAMES:
        assert name in tasks, f"Task '{name}' not found in registry"


def test_sumo_controller_import():
    module = importlib.import_module("sumo.controller")
    assert hasattr(module, "Controller")
    assert hasattr(module, "ControllerConfig")


def test_sumo_run_mpc_import():
    module = importlib.import_module("sumo.run_mpc.run_mpc")
    assert hasattr(module, "run_mpc")
    assert hasattr(module, "RunMPCConfig")


def test_sumo_dora_imports():
    from sumo.app.dora.controller_node import ControllerNode
    from sumo.app.dora.simulation_node import G1Simulation, SimulationNode

    assert ControllerNode is not None
    assert SimulationNode is not None
    assert G1Simulation is not None


def test_prefer_active_env_libs(monkeypatch, tmp_path):
    import sumo

    prefix = tmp_path / "pixi-env"
    lib_dir = prefix / "lib"
    lib_dir.mkdir(parents=True)

    monkeypatch.setenv("CONDA_PREFIX", str(prefix))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    monkeypatch.delenv("DYLD_FALLBACK_LIBRARY_PATH", raising=False)

    sumo._prefer_active_env_libs()
    assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(lib_dir)

    sumo._prefer_active_env_libs()
    assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep).count(str(lib_dir)) == 1


def test_simulation_backend_resolution_is_lazy():
    from sumo.app.dora import simulation_node as simulation_module

    # G1Simulation is defined locally in sumo
    assert hasattr(simulation_module, "G1Simulation")
    # SimulationNode overrides _init_sim for G1 and lazy policy loading
    assert hasattr(simulation_module.SimulationNode, "_init_sim")


def test_sumo_task_filtering():
    from sumo.tasks import SUMO_TASK_NAMES, get_sumo_registered_tasks

    available_tasks = get_sumo_registered_tasks()
    assert set(SUMO_TASK_NAMES) <= set(available_tasks)
    assert "cartpole" not in available_tasks
    assert "cylinder_push" not in available_tasks


@pytest.mark.g1_extensions
def test_g1_extensions_import():
    from g1_extensions import G1Rollout, sim_g1

    assert G1Rollout is not None
    assert sim_g1 is not None
