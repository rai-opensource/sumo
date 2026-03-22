# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Controller with G1 rollout backend support."""

from __future__ import annotations

from typing import Literal

import numpy as np
from judo.controller.controller import Controller, ControllerConfig, make_spline
from judo.controller.controller import make_controller as _judo_make_controller
from omegaconf import DictConfig

from sumo.utils.mujoco import RolloutBackend as SumoRolloutBackend
from sumo.utils.mujoco import make_model_data_pairs


class G1RolloutBackendAdapter:
    """Adapter that exposes the local G1 rollout backend through judo's interface."""

    def __init__(
        self,
        model,
        num_threads: int,
        cutoff_time: float = 0.2,
    ) -> None:
        self.model = model
        self.num_threads = num_threads
        self.cutoff_time = cutoff_time
        self._backend = SumoRolloutBackend(
            num_threads=num_threads,
            backend="mujoco_g1",
            task_to_sim_ctrl=lambda controls: controls,
            cutoff_time=cutoff_time,
        )
        self._model_data_pairs = make_model_data_pairs(model, num_threads)

    def rollout(
        self,
        x0: np.ndarray,
        controls: np.ndarray,
        last_policy_output: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        states, sensors = self._backend.rollout(self._model_data_pairs, x0, controls)
        return states, sensors, None

    def update(self, num_threads: int) -> None:
        self.num_threads = num_threads
        self._model_data_pairs = make_model_data_pairs(self.model, num_threads)
        self._backend.update(num_threads, self.cutoff_time)


def make_controller(
    init_task: str,
    init_optimizer: str,
    task_registration_cfg: DictConfig | None = None,
    optimizer_registration_cfg: DictConfig | None = None,
    rollout_backend: Literal["mujoco"] = "mujoco",
) -> Controller:
    """Make a controller with G1 backend support."""
    return _judo_make_controller(
        init_task=init_task,
        init_optimizer=init_optimizer,
        task_registration_cfg=task_registration_cfg,
        optimizer_registration_cfg=optimizer_registration_cfg,
        rollout_backend=rollout_backend,
        controller_kwargs={"custom_rollout_backends": {"mujoco_g1": G1RolloutBackendAdapter}},
    )


__all__ = ["Controller", "ControllerConfig", "make_controller", "make_spline"]
