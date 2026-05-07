# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from judo.app.dora.visualization_node import VisualizationNode as JudoVisualizationNode

import sumo.controller  # noqa: F401 -- register controller/optimizer overrides
import sumo.tasks  # noqa: F401 -- register all sumo tasks
from sumo.tasks import get_sumo_registered_tasks


class VisualizationNode(JudoVisualizationNode):
    """Visualization node with sumo-only task choices."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("available_tasks", get_sumo_registered_tasks())
        super().__init__(**kwargs)


__all__ = ["VisualizationNode"]
