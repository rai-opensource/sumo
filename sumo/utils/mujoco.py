# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""G1 rollout backend using sumo's local C++ extensions."""

import numpy as np
from judo.utils.mj_rollout_backend import make_model_data_pairs
from judo.utils.rollout_backend import RolloutBackend
from mujoco import MjModel

from sumo.utils.extensions import require_g1_extensions

# Re-export for convenience
__all__ = ["G1RolloutBackend", "make_model_data_pairs"]


class G1RolloutBackend(RolloutBackend):
    """Rollout backend for G1 tasks using C++ G1Rollout extension."""

    def __init__(
        self,
        model: MjModel,
        num_threads: int,
        cutoff_time: float = 0.2,
    ) -> None:
        self.model = model
        self.num_threads = num_threads
        self.cutoff_time = cutoff_time
        g1_extensions = require_g1_extensions()
        self._rollout_obj = g1_extensions.G1Rollout(nthread=num_threads, cutoff_time=cutoff_time)
        self._models, self._datas = make_model_data_pairs(model, num_threads)

    def rollout(
        self,
        x0: np.ndarray,
        controls: np.ndarray,
        last_policy_output: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        """Conduct parallel rollouts using the G1 C++ backend."""
        x0_batched = np.tile(x0, (len(self._models), 1))
        out_states, out_sensors = self._rollout_obj.rollout(self._models, self._datas, x0_batched, controls)
        return np.array(out_states), np.array(out_sensors), None

    def update(self, num_threads: int) -> None:
        """Update the number of threads."""
        self.num_threads = num_threads
        self._rollout_obj.close()
        g1_extensions = require_g1_extensions()
        self._rollout_obj = g1_extensions.G1Rollout(nthread=num_threads, cutoff_time=self.cutoff_time)
        self._models, self._datas = make_model_data_pairs(self.model, num_threads)
