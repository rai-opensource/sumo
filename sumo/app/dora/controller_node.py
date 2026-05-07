# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from judo.app.dora.controller_node import ControllerNode as JudoControllerNode

import sumo.controller  # noqa: F401 -- register controller/optimizer overrides
import sumo.tasks  # noqa: F401 -- register all sumo tasks
from sumo.controller import make_controller


class ControllerNode(JudoControllerNode):
    """Controller node that uses sumo's make_controller for G1 backend support."""

    def __init__(self, init_task: str = "spot_box_push", **kwargs) -> None:
        super().__init__(
            init_task=init_task,
            make_controller_fn=make_controller,
            **kwargs,
        )


__all__ = ["ControllerNode"]
