# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Headless MPC runner for benchmarking tasks."""

import inspect
import os
from dataclasses import dataclass

import h5py
import numpy as np
import tyro
from judo.app.structs import MujocoState
from judo.optimizers import get_registered_optimizers
from judo.tasks import get_registered_tasks
from tqdm import tqdm

import sumo.controller  # noqa: F401 -- register controller/optimizer overrides
import sumo.tasks  # noqa: F401 -- register all sumo tasks
from sumo.app.dora.g1_simulation import G1Simulation
from sumo.controller import Controller, ControllerConfig
from sumo.utils.extensions import require_g1_extensions, require_mujoco_extensions
from sumo.utils.mujoco import G1RolloutBackend


@dataclass
class RunMPCConfig:
    """Config class for MPC runs."""

    init_task: str = "spot_box_push"
    init_optimizer: str = "cem"
    visualize: bool = True
    num_episodes: int = 2
    episode_length_s: float = 10.0
    save_results: bool = True
    output_dir: str = "run_mpc/results"
    verbose: bool = False
    viz_dt: float = 0.02
    # Data recording options
    record_all_data: bool = False
    record_qvel: bool = False
    record_xpos: bool = False
    record_xquat: bool = False
    record_ctrl: bool = False
    record_sensordata: bool = False
    record_mocap: bool = False
    record_traces: bool = False
    # Optimizer rollout recording options
    record_rollouts: bool = True
    record_rollout_controls: bool = False
    record_rollout_sensors: bool = False


# ---------------------------------------------------------------------------
# Simulation backend resolution
# ---------------------------------------------------------------------------


def _create_sim(task_name: str):
    """Create the right simulation backend for a task."""
    task_entry = get_registered_tasks()[task_name]
    simulation_backend = task_entry.simulation_backend

    if simulation_backend == "mujoco_g1":
        require_g1_extensions()
        return G1Simulation(init_task=task_name)

    if simulation_backend == "mujoco_hierarchical":
        require_mujoco_extensions()
        from judo.simulation import get_simulation_backend

        return get_simulation_backend("mujoco_hierarchical")(init_task=task_name)

    from judo.simulation import get_simulation_backend

    return get_simulation_backend(simulation_backend)(init_task=task_name)


def _make_condition_checker(method):
    """Build a callable for success/failure that handles both G1 and Spot signatures."""
    if method is None:
        return None
    if "config" in inspect.signature(method).parameters:
        return lambda model, data, config: method(model, data, config)
    return lambda model, data, config: method(model, data)


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------


def run_single_episode(config, task, controller, sim, viser_model=None, episode_idx=0):
    """Run one episode, return recorded data."""
    sim_dt = task.sim_model.opt.timestep
    plan_dt = 1.0 / controller.controller_cfg.control_freq
    ctrl_dt = task.model.opt.timestep
    num_steps = int(config.episode_length_s / sim_dt) + 1

    steps_per_plan = max(1, int(plan_dt / sim_dt))
    steps_per_ctrl = max(1, int(ctrl_dt / sim_dt))
    steps_per_record = max(1, int(config.viz_dt / sim_dt))

    # Reset
    np.random.seed(episode_idx)
    task.reset()
    controller.reset()
    task.data.time = 0.0

    check_success = _make_condition_checker(getattr(task, "success", None))
    check_failure = _make_condition_checker(getattr(task, "failure", None))
    should_record = lambda field: config.record_all_data or getattr(config, f"record_{field}", False)

    data = {
        "rewards": [],
        "qpos_traj": [],
        "time_traj": [],
        "success": False,
        "failure": False,
        "length": 0.0,
    }
    if should_record("qvel"):
        data["qvel_traj"] = []
    if should_record("xpos"):
        data["xpos_traj"] = []
    if should_record("xquat"):
        data["xquat_traj"] = []
    if should_record("ctrl"):
        data["ctrl_traj"] = []
    if should_record("sensordata"):
        data["sensordata_traj"] = []
    if should_record("mocap"):
        data["mocap_pos_traj"] = []
        data["mocap_quat_traj"] = []
    if should_record("traces"):
        data["traces_traj"] = []
    if should_record("rollouts"):
        data["rollout_states"] = []
        data["rollout_rewards"] = []
    if should_record("rollout_controls"):
        data["rollout_controls"] = []
    if should_record("rollout_sensors"):
        data["rollout_sensors"] = []

    current_action = None
    curr_time = 0.0

    for step in tqdm(range(num_steps), desc=f"Episode {episode_idx + 1}/{config.num_episodes}", leave=False):
        curr_time = step * sim_dt

        # Plan
        if step % steps_per_plan == 0:
            controller.update_states(
                MujocoState(
                    time=curr_time,
                    qpos=np.array(task.data.qpos),
                    qvel=np.array(task.data.qvel),
                    mocap_pos=np.array(task.data.mocap_pos),
                    mocap_quat=np.array(task.data.mocap_quat),
                    sim_metadata={},
                )
            )
            controller.update_action()
            data["rewards"].append(float(controller.rewards.max()))

            if should_record("rollouts"):
                data["rollout_states"].append(np.array(controller.states))
                data["rollout_rewards"].append(np.array(controller.rewards))
            if should_record("rollout_controls"):
                data["rollout_controls"].append(np.array(controller.rollout_controls))
            if should_record("rollout_sensors"):
                data["rollout_sensors"].append(np.array(controller.sensors))

        # Control
        if step % steps_per_ctrl == 0:
            current_action = controller.action(curr_time)

        # Step simulation
        if current_action is not None:
            sim.step(current_action)
        controller.system_metadata = task.get_sim_metadata()
        task.post_sim_step()

        # Check termination
        sim_model = task.sim_model
        if check_success is not None:
            data["success"] = data["success"] or check_success(sim_model, task.data, task.config)
        if check_failure is not None:
            data["failure"] = data["failure"] or check_failure(sim_model, task.data, task.config)

        # Update visualization at real-time rate
        if viser_model is not None and step % steps_per_record == 0:
            viser_model.set_data(task.data)
            # time.sleep(config.viz_dt)

        # Record trajectory
        if step % steps_per_record == 0:
            data["time_traj"].append(curr_time)
            data["qpos_traj"].append(np.array(task.data.qpos))
            if should_record("qvel"):
                data["qvel_traj"].append(np.array(task.data.qvel))
            if should_record("xpos"):
                data["xpos_traj"].append(np.array(task.data.xpos))
            if should_record("xquat"):
                data["xquat_traj"].append(np.array(task.data.xquat))
            if should_record("ctrl"):
                data["ctrl_traj"].append(np.array(task.data.ctrl))
            if should_record("sensordata"):
                data["sensordata_traj"].append(np.array(task.data.sensordata))
            if should_record("mocap"):
                data["mocap_pos_traj"].append(np.array(task.data.mocap_pos))
                data["mocap_quat_traj"].append(np.array(task.data.mocap_quat))
            if should_record("traces") and controller.traces is not None:
                data["traces_traj"].append(np.array(controller.traces))

    data["length"] = curr_time
    for key, val in data.items():
        if isinstance(val, list):
            data[key] = np.asarray(val) if val else np.empty((0,))
    return data


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_mpc(config: RunMPCConfig) -> list[dict]:
    """Set up and run MPC for multiple episodes."""
    task_dict = get_registered_tasks()
    if config.init_task not in task_dict:
        raise ValueError(f"Task '{config.init_task}' is not registered.")

    # Create simulation backend (handles G1, Spot locomotion policy, plain MuJoCo)
    sim = _create_sim(config.init_task)
    task = sim.task

    # Create optimizer
    optimizer_dict = get_registered_optimizers()
    if config.init_optimizer not in optimizer_dict:
        raise ValueError(f"Optimizer '{config.init_optimizer}' is not registered.")
    optimizer_cls, optimizer_config_cls = optimizer_dict[config.init_optimizer]
    optimizer_config = optimizer_config_cls()
    optimizer_config.set_override(config.init_task)
    optimizer = optimizer_cls(optimizer_config, task.nu)

    # Create controller
    controller_config = ControllerConfig()
    controller_config.set_override(config.init_task)
    controller = Controller(controller_config, task, optimizer, rollout_backend_registry={"mujoco_g1": G1RolloutBackend})

    # Set up visualization
    viser_model = None
    if config.visualize:
        from dataclasses import fields as dc_fields

        import viser
        from judo.visualizers.model import ViserMjModel

        server = viser.ViserServer()
        viser_model = ViserMjModel(server, task.spec, geom_exclude_substring="collision")
        viser_model.set_data(task.data)

        # Render goal position markers from task config metadata
        for f in dc_fields(task.config):
            ui_cfg = f.metadata.get("ui_array_config") if f.metadata else None
            vis = ui_cfg.get("vis") if ui_cfg else None
            if vis is None:
                continue
            goal_val = np.asarray(getattr(task.config, f.name))
            xyz_indices = vis["xyz_vis_indices"]
            xyz_defaults = vis.get("xyz_vis_defaults", (0.0, 0.0, 0.0))
            position = np.array(
                [
                    goal_val[xyz_indices[i]] if xyz_indices[i] is not None else xyz_defaults[i]
                    for i in range(len(xyz_indices))
                ]
            )
            server.scene.add_icosphere(vis["name"], radius=0.15, color=(0.0, 0.0, 1.0), position=position)

        print("Visualizer running at: http://localhost:8080")

    # Run episodes
    all_episodes = []
    for i in range(config.num_episodes):
        episode = run_single_episode(config, task, controller, sim, viser_model=viser_model, episode_idx=i)
        all_episodes.append(episode)

    # Print summary
    avg_reward = np.mean([np.mean(ep["rewards"]) for ep in all_episodes])
    num_successes = sum(ep["success"] for ep in all_episodes)
    num_failures = sum(ep["failure"] for ep in all_episodes)
    avg_length = np.mean([ep["length"] for ep in all_episodes])

    print(f"\n{'=' * 60}")
    print(f"Task: {config.init_task}  Optimizer: {config.init_optimizer}")
    print(f"Episodes: {config.num_episodes}  Length: {avg_length:.1f}s")
    print(
        f"Avg Reward: {avg_reward:.2f}  Success: {num_successes}/{config.num_episodes}  Failure: {num_failures}/{config.num_episodes}"
    )
    print(f"{'=' * 60}")

    # Save results
    if config.save_results:
        _save_results(config, all_episodes, task)

    return all_episodes


def _save_results(config: RunMPCConfig, all_episodes: list[dict], task) -> None:
    """Save episode data to HDF5."""
    output_dir = os.path.join(config.output_dir, config.init_task)
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, "trajectories.h5")

    with h5py.File(h5_path, "w") as f:
        f.attrs["task"] = config.init_task
        f.attrs["optimizer"] = config.init_optimizer
        f.attrs["num_episodes"] = config.num_episodes
        f.attrs["episode_length_s"] = config.episode_length_s
        f.attrs["sim_dt"] = task.sim_model.opt.timestep

        task_group = f.create_group(config.init_task)
        opt_group = task_group.create_group(config.init_optimizer)

        for i, ep in enumerate(all_episodes):
            g = opt_group.create_group(f"episode_{i}")
            g.attrs["success"] = ep["success"]
            g.attrs["failure"] = ep["failure"]
            g.attrs["length"] = ep["length"]
            for key, val in ep.items():
                if isinstance(val, np.ndarray) and val.size > 0:
                    g.create_dataset(key, data=val)

    print(f"Results saved to: {h5_path}")


if __name__ == "__main__":
    run_mpc(tyro.cli(RunMPCConfig))
