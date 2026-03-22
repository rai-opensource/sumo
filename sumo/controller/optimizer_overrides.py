# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from judo.config import set_config_overrides
from judo.optimizers.base import OptimizerConfig
from judo.optimizers.cem import CrossEntropyMethodConfig
from judo.optimizers.mppi import MPPIConfig
from judo.optimizers.ps import PredictiveSamplingConfig

from sumo.tasks import SPOT_TASK_NAMES

_SPOT_OPTIMIZER_BASE = {
    "num_rollouts": 24,
    "num_nodes": 3,
    "use_noise_ramp": True,
    "noise_ramp": 3.5,
}


def _set_spot_optimizer_overrides(task_name: str) -> None:
    """Sets the default optimizer config overrides for a Spot task."""
    set_config_overrides(task_name, OptimizerConfig, _SPOT_OPTIMIZER_BASE)
    set_config_overrides(task_name, PredictiveSamplingConfig, _SPOT_OPTIMIZER_BASE)
    set_config_overrides(task_name, CrossEntropyMethodConfig, {**_SPOT_OPTIMIZER_BASE, "num_elites": 3})
    set_config_overrides(task_name, MPPIConfig, _SPOT_OPTIMIZER_BASE)


def set_default_spot_optimizer_overrides() -> None:
    """Sets the default task-specific optimizer config overrides for all Spot tasks."""
    for task_name in SPOT_TASK_NAMES:
        _set_spot_optimizer_overrides(task_name)
