# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import warnings
from pathlib import Path

import hydra
from dora_utils.launch.run import run
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore

# Importing judo.cli registers process cleanup handlers (atexit, signal, Popen wrapper)
from judo.cli import (
    CONFIG_PATH as JUDO_CONFIG_PATH,
)
from judo.cli import (
    _force_cleanup,
    _warm_caches,
)
from omegaconf import DictConfig

# suppress hydra warning
warnings.filterwarnings(
    "ignore",
    message=r".*Defaults list is missing `_self_`.*",
    category=UserWarning,
    module="hydra._internal.defaults_list",
)

SUMO_CONFIG_PATH = (Path(__file__).parent / "configs").resolve()


@hydra.main(config_path=str(SUMO_CONFIG_PATH), config_name="sumo_default", version_base="1.3")
def _main(cfg: DictConfig) -> None:
    try:
        run(cfg)
    except (KeyboardInterrupt, SystemExit):
        _force_cleanup()


def app() -> None:
    """Entry point for the sumo CLI."""
    import sumo.controller  # noqa: F401 -- register controller/optimizer overrides
    import sumo.tasks  # noqa: F401 -- register all tasks

    _warm_caches()

    # Store judo's default config in ConfigStore so sumo_default.yaml can inherit it
    cs = ConfigStore.instance()
    with initialize_config_dir(config_dir=str(JUDO_CONFIG_PATH), version_base="1.3"):
        default_cfg = compose(config_name="judo_dora_default")
        cs.store("judo_dora", default_cfg)

    _main()
