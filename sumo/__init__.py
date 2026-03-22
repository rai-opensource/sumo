# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PACKAGE_ROOT / "models"


def _prepend_env_path(env_var: str, path: Path) -> None:
    """Prepend a path-like environment variable without duplicating entries."""
    path_str = str(path)
    current = os.environ.get(env_var, "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if path_str in entries:
        return
    os.environ[env_var] = os.pathsep.join([path_str, *entries]) if entries else path_str


def _prefer_active_env_libs() -> None:
    """Prefer shared libraries from the active Pixi/Conda environment.

    Some dependencies import `pyarrow` before `numpy`, which can otherwise bind
    against an older system `libstdc++` and break later NumPy imports.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    lib_dir = Path(conda_prefix) / "lib"
    if not lib_dir.is_dir():
        return

    _prepend_env_path("LD_LIBRARY_PATH", lib_dir)
    if sys.platform == "darwin":
        _prepend_env_path("DYLD_FALLBACK_LIBRARY_PATH", lib_dir)


_prefer_active_env_libs()
