# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from judo.tasks.spot.spot_box_push import SpotBoxPush as _JudoSpotBoxPush
from judo.tasks.spot.spot_box_push import SpotBoxPushConfig

from sumo.tasks.spot.spot_base import SpotAssetMixin


class SpotBoxPush(SpotAssetMixin, _JudoSpotBoxPush):
    """Public judo-rai SpotBoxPush with Sumo asset resolution."""


__all__ = ["SpotBoxPush", "SpotBoxPushConfig"]
