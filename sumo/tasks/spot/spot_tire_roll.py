# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from judo.tasks.spot.spot_tire_roll import SpotTireRoll as _JudoSpotTireRoll
from judo.tasks.spot.spot_tire_roll import SpotTireRollConfig

from sumo.tasks.spot.spot_base import SpotAssetMixin


class SpotTireRoll(SpotAssetMixin, _JudoSpotTireRoll):
    """Public judo-rai SpotTireRoll with Sumo asset resolution."""


__all__ = ["SpotTireRoll", "SpotTireRollConfig"]
