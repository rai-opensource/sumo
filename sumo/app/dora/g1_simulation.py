# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""G1 simulation backend using sumo's local C++ policy wrapper."""

from typing import Callable

import numpy as np
from judo.simulation.base import Simulation
from mujoco import MjData, MjModel

from sumo.utils.extensions import require_g1_extensions


class SimBackendG1:
    """Single-step simulation backend for G1 using the C++ policy wrapper."""

    def __init__(self, task_to_sim_ctrl: Callable) -> None:
        self.task_to_sim_ctrl = task_to_sim_ctrl
        self.previous_policy_output = np.zeros(29)

    def sim(self, sim_model: MjModel, sim_data: MjData, sim_controls: np.ndarray) -> None:
        """Conduct a simulation step."""
        g1_extensions = require_g1_extensions()
        x0 = np.concatenate([sim_data.qpos, sim_data.qvel])
        controls = self.task_to_sim_ctrl(sim_controls).flatten()
        self.previous_policy_output = g1_extensions.sim_g1(
            sim_model,
            sim_data,
            x0,
            controls,
            self.previous_policy_output,
        )


class G1Simulation(Simulation):
    """Simulation backend for G1 tasks using the local C++ policy wrapper."""

    def __init__(self, **kwargs) -> None:
        self._sim_backend: SimBackendG1 | None = None
        super().__init__(**kwargs)

    def set_task(self, task_name: str) -> None:
        super().set_task(task_name)
        self._sim_backend = SimBackendG1(task_to_sim_ctrl=self.task.task_to_sim_ctrl)

    def step(self, command: np.ndarray) -> None:
        if self.paused:
            return
        assert self._sim_backend is not None
        self.task.pre_sim_step()
        self._sim_backend.sim(self.task.sim_model, self.task.data, command)
        self.task.post_sim_step()
