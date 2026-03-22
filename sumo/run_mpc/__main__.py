# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import tyro

from sumo.run_mpc.run_mpc import RunMPCConfig, run_mpc

run_mpc(tyro.cli(RunMPCConfig))
