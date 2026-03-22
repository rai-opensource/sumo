# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from judo.tasks.spot.spot_tire_upright import SpotTireUpright as _JudoSpotTireUpright
from judo.tasks.spot.spot_tire_upright import SpotTireUprightConfig

from sumo.tasks.spot.spot_base import SpotAssetMixin


class SpotTireUpright(SpotAssetMixin, _JudoSpotTireUpright):
    """Public judo-rai SpotTireUpright with Sumo asset resolution."""


__all__ = ["SpotTireUpright", "SpotTireUprightConfig"]
