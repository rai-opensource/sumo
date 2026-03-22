# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""G1 simulation backend using sumo's local C++ policy wrapper."""

import numpy as np
from judo.simulation.base import Simulation

from sumo.utils.mujoco import SimBackendG1


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
