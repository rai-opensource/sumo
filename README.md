# Sumo 🥋

<p align="center">
  <a href="https://arxiv.org/abs/2604.08508"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper"></a>
  <a href="https://sumo.rai-inst.com"><img src="https://img.shields.io/badge/Project-Website-4c84f3.svg" alt="Project Website"></a>
  <a href="https://www.youtube.com/watch?v=eKUSpB9G7Rk"><img src="https://img.shields.io/badge/Video-Watch-ff0000.svg" alt="Demo Video"></a>
</p>

<p align="center">
  <img src="asset/sumo.gif" alt="task dropdown" width="640">
</p>

`sumo`🥋 is a research codebase released along with our [paper](https://arxiv.org/abs/2604.08508) on whole-body loco-manipulation.

## Install

Requires [pixi](https://pixi.sh).

```bash
# Install the default app environment
pixi install

# Build the local g1_extensions extension (required for G1 tasks)
# and judo's mujoco_extensions (required for Spot tasks)
pixi run build

```

`pixi run sumo` assumes both native extensions above have already been built.

If you prefer named environments, `pixi install -e dev` is equivalent for this repo,
and the corresponding task form is `pixi run -e dev ...`.

### Cleaning the build

If native extensions misbehave (e.g. after a mujoco / judo / pybind11 version
bump, or if you see odd shape mismatches between Python and the C++ rollout
backend), wipe the build artifacts before rebuilding:

```bash
# Remove C++ build dirs, deployed .so files, and __pycache__ directories
pixi run clean-build
pixi run build

# Full reset: also removes .judo-src/ and .pixi/ (forces fresh clone + reinstall)
pixi run clean-all
pixi run build   # auto-runs `pixi install` first
```

## Run

```bash
# Launch simulation + visualizer with default task
pixi run sumo

# Specify a task
pixi run sumo task=spot_box_push

# Specify a task and other hydra options (e.g. optimizer)
pixi run sumo task=g1_box optimizer=mppi
```

## Headless MPC Runner

`run_mpc` runs MPC episodes without the full GUI, useful for benchmarking and data collection.

```bash
# Run with defaults (spot_box_push, CEM optimizer, 2 episodes)
pixi run python -m sumo.run_mpc

# Specify task and options
pixi run python -m sumo.run_mpc --init-task=g1_door --init-optimizer=cem --num-episodes=10

# With visualization
pixi run python -m sumo.run_mpc --init-task=g1_box --visualize

# Disable saving results
pixi run python -m sumo.run_mpc --init-task=spot_box_push --no-save-results

# Record all rollout data
pixi run python -m sumo.run_mpc --init-task=g1_door --record-all-data
```

Results are saved to `run_mpc/results/` by default as HDF5 files. Use `--help` to see all options.

## Tests

```bash
pixi run pytest tests/ -v
```

Disclaimer: this code is released accompanying a paper submission only.

## Reference

If you find this codebase helpful, please consider citing our paper.

```bibtex
@article{zhang2026sumo,
  title = {Sumo: Dynamic and Generalizable Whole-Body Loco-Manipulation},
  author = {Zhang, John Z. and Sorokin, Maks and Br{\"u}digam, Jan and Hung, Brandon and Phillips, Stephen and Yershov, Dmitry and Niroui, Farzad and Zhao, Tong and Fermoselle, Leonor and Zhu, Xinghao and Cao, Chao and Ta, Duy and Pang, Tao and Wang, Jiuguang and Culbertson, Preston and Manchester, Zachary and Le Cl\'eac'h, Simon},
  journal = {arXiv preprint arXiv:2604.08508},
  year = {2026},
  url = {https://arxiv.org/abs/2604.08508}
}
```
