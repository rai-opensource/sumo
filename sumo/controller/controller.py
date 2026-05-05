# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Controller with G1 rollout backend support."""

from __future__ import annotations

from typing import Literal

from judo.controller.controller import Controller, ControllerConfig, make_spline
from judo.controller.controller import make_controller as _judo_make_controller
from omegaconf import DictConfig

from sumo.utils.mujoco import G1RolloutBackend


def make_controller(
    init_task: str,
    init_optimizer: str,
    task_registration_cfg: DictConfig | None = None,
    optimizer_registration_cfg: DictConfig | None = None,
    rollout_backend: Literal["mujoco", "mujoco_hierarchical", "mujoco_g1"] = "mujoco",
) -> Controller:
    """Make a controller with G1 backend support."""
    return _judo_make_controller(
        init_task=init_task,
        init_optimizer=init_optimizer,
        task_registration_cfg=task_registration_cfg,
        optimizer_registration_cfg=optimizer_registration_cfg,
        rollout_backend=rollout_backend,
        rollout_backend_registry={"mujoco_g1": G1RolloutBackend},
    )


__all__ = ["Controller", "ControllerConfig", "make_controller", "make_spline"]
