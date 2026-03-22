# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import time
from copy import deepcopy
from typing import Callable, Literal

import numpy as np
from mujoco import MjData, MjModel, mj_step
from mujoco.rollout import Rollout

from sumo.utils.extensions import require_g1_extensions


def make_model_data_pairs(model: MjModel, num_pairs: int) -> list[tuple[MjModel, MjData]]:
    """Create model/data pairs for mujoco threaded rollout."""
    models = [deepcopy(model) for _ in range(num_pairs)]
    datas = [MjData(m) for m in models]
    model_data_pairs = list(zip(models, datas, strict=True))
    return model_data_pairs


class RolloutBackend:
    """The backend for conducting multithreaded rollouts."""

    def __init__(
        self,
        num_threads: int,
        backend: Literal["mujoco", "mujoco_g1"],
        task_to_sim_ctrl: Callable,
        cutoff_time: float = 0.2,
    ) -> None:
        """Initialize the backend with a number of threads."""
        self.backend = backend
        self.num_threads = num_threads
        self.cutoff_time = cutoff_time
        if self.backend == "mujoco":
            self.setup_mujoco_backend(num_threads)
        elif self.backend == "mujoco_g1":
            self.setup_mujoco_g1_backend(num_threads, cutoff_time)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        self.task_to_sim_ctrl = task_to_sim_ctrl

    def setup_mujoco_backend(self, num_threads: int) -> None:
        """Setup the mujoco backend."""
        if self.backend == "mujoco":
            self.rollout_obj = Rollout(nthread=num_threads)
            self.rollout_func = lambda m, d, x0, u: self.rollout_obj.rollout(m, d, x0, u)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def setup_mujoco_g1_backend(self, num_threads: int, cutoff_time: float) -> None:
        """Setup the mujoco_g1 backend with G1Rollout object."""
        g1_extensions = require_g1_extensions()
        self.rollout_obj = g1_extensions.G1Rollout(nthread=num_threads, cutoff_time=cutoff_time)
        self.rollout_func = lambda m, d, x0, u: self.rollout_obj.rollout(m, d, x0, u)

    def rollout(
        self,
        model_data_pairs: list[tuple[MjModel, MjData]],
        x0: np.ndarray,
        controls: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Conduct a rollout depending on the backend."""
        # unpack models into a list of models and data
        ms, ds = zip(*model_data_pairs, strict=True)
        ms = list(ms)
        ds = list(ds)

        # getting shapes
        nq = ms[0].nq
        nv = ms[0].nv
        nu = ms[0].nu

        # the state passed into mujoco's rollout function includes the time
        # shape = (num_rollouts, num_states + 1)
        x0_batched = np.tile(x0, (len(ms), 1))
        full_states = np.concatenate([time.time() * np.ones((len(ms), 1)), x0_batched], axis=-1)
        processed_controls = self.task_to_sim_ctrl(controls)
        assert full_states.shape[-1] == nq + nv + 1
        assert full_states.ndim == 2
        assert processed_controls.ndim == 3
        assert processed_controls.shape[0] == full_states.shape[0]

        # rollout
        if self.backend == "mujoco":
            _states, _out_sensors = self.rollout_func(ms, ds, full_states, processed_controls)
            out_states = np.array(_states)[..., 1:]  # remove time from state
            out_sensors = np.array(_out_sensors)
            return out_states, out_sensors
        elif self.backend == "mujoco_g1":
            out_states, out_sensors = self.rollout_func(ms, ds, x0_batched, processed_controls)
            return np.array(out_states), np.array(out_sensors)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def update(self, num_threads: int, cutoff_time: float | None = None) -> None:
        """Update the backend with a new number of threads and optionally cutoff time."""
        self.num_threads = num_threads
        if cutoff_time is not None:
            self.cutoff_time = cutoff_time
        if self.backend == "mujoco":
            self.rollout_obj.close()
            self.setup_mujoco_backend(num_threads)
        elif self.backend == "mujoco_g1":
            self.rollout_obj.close()
            self.setup_mujoco_g1_backend(num_threads, self.cutoff_time)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")


class SimBackend:
    """The backend for conducting simulation."""

    def __init__(self, task_to_sim_ctrl: Callable) -> None:
        """Initialize the backend."""
        self.task_to_sim_ctrl = task_to_sim_ctrl

    def sim(self, sim_model: MjModel, sim_data: MjData, sim_controls: np.ndarray) -> None:
        """Conduct a simulation step."""
        sim_data.ctrl[:] = self.task_to_sim_ctrl(sim_controls)
        mj_step(sim_model, sim_data)


class SimBackendG1:
    def __init__(self, task_to_sim_ctrl: Callable) -> None:
        self.task_to_sim_ctrl = task_to_sim_ctrl
        self.previous_policy_output = np.zeros((29))

    def sim(self, sim_model: MjModel, sim_data: MjData, sim_controls: np.ndarray) -> None:
        """Conduct a simulation step."""
        g1_extensions = require_g1_extensions()
        x0 = np.concatenate([sim_data.qpos, sim_data.qvel])
        controls = self.task_to_sim_ctrl(sim_controls)
        controls = controls.flatten()
        self.previous_policy_output = g1_extensions.sim_g1(
            sim_model,
            sim_data,
            x0,
            controls,
            self.previous_policy_output,
        )
