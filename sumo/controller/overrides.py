# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from judo.config import set_config_overrides
from judo.controller.controller import ControllerConfig

from sumo.tasks import SPOT_TASK_NAMES


def set_default_g1_box_overrides() -> None:
    """Sets the default task-specific controller config overrides for the g1 box task."""
    set_config_overrides(
        "g1_box",
        ControllerConfig,
        {
            "horizon": 2.5,
            "spline_order": "cubic",
        },
    )


def set_default_g1_door_overrides() -> None:
    """Sets the default task-specific controller config overrides for the g1 door task."""
    set_config_overrides(
        "g1_door",
        ControllerConfig,
        {
            "horizon": 2.5,
            "spline_order": "cubic",
        },
    )


def set_default_g1_chair_push_overrides() -> None:
    """Sets the default task-specific controller config overrides for the g1 chair push task."""
    set_config_overrides(
        "g1_chair_push",
        ControllerConfig,
        {
            "horizon": 2.5,
            "spline_order": "cubic",
        },
    )


def set_default_g1_table_push_overrides() -> None:
    """Sets the default task-specific controller config overrides for the g1 table push task."""
    set_config_overrides(
        "g1_table_push",
        ControllerConfig,
        {
            "horizon": 2.5,
            "spline_order": "cubic",
        },
    )


def set_default_spot_overrides() -> None:
    """Sets the default task-specific controller config overrides for all Spot tasks."""
    for task_name in SPOT_TASK_NAMES:
        set_config_overrides(
            task_name,
            ControllerConfig,
            {
                "horizon": 2.0,
            },
        )
