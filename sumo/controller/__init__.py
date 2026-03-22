# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from sumo.controller.controller import Controller, ControllerConfig, make_controller
from sumo.controller.optimizer_overrides import set_default_spot_optimizer_overrides
from sumo.controller.overrides import (
    set_default_g1_box_overrides,
    set_default_g1_chair_push_overrides,
    set_default_g1_door_overrides,
    set_default_g1_table_push_overrides,
    set_default_spot_overrides,
)

# Register overrides on import
set_default_g1_box_overrides()
set_default_g1_door_overrides()
set_default_g1_chair_push_overrides()
set_default_g1_table_push_overrides()
set_default_spot_overrides()
set_default_spot_optimizer_overrides()

__all__ = ["Controller", "ControllerConfig", "make_controller"]
