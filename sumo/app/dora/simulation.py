# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from judo.app.dora.simulation import SimulationNode as JudoSimulationNode

import sumo.tasks  # noqa: F401 -- register all sumo tasks
from sumo.app.dora.g1_simulation import G1Simulation


class SimulationNode(JudoSimulationNode):
    """Simulation node with G1 backend support."""

    def __init__(self, init_task: str = "spot_box_push", **kwargs) -> None:
        kwargs.setdefault("custom_backends", {"mujoco_g1": G1Simulation})
        super().__init__(init_task=init_task, **kwargs)


__all__ = ["G1Simulation", "SimulationNode"]
