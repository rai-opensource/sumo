# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from importlib import import_module
from types import ModuleType


def require_g1_extensions() -> ModuleType:
    """Import the local G1 native extension with a repo-specific error."""
    try:
        return import_module("g1_extensions")
    except ImportError as exc:
        raise ImportError(
            "G1 tasks require the local `g1_extensions` package.\n"
            "Install the dev environment and build native extensions with:\n"
            "  pixi install -e dev\n"
            "  pixi run build"
        ) from exc


def require_mujoco_extensions() -> ModuleType:
    """Import Judo's MuJoCo policy extension with a repo-specific error."""
    try:
        return import_module("mujoco_extensions.policy_rollout")
    except ImportError as exc:
        raise ImportError(
            "Spot tasks require Judo's `mujoco_extensions` package.\n"
            "Install the dev environment and build native extensions with:\n"
            "  pixi install -e dev\n"
            "  pixi run build-judo-ext\n"
            "For the full sumo app, G1 tasks also require:\n"
            "  pixi run build"
        ) from exc
