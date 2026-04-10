# Sumo 🥋
Disclaimer: this code is released accompanying a paper submission only. The paper itself can be found [here]().

`sumo`🥋 is a research codebase for whole-body loco-manipulation built on
[`judo`🥋](https://github.com/bdaiinstitute/judo).

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

## Reference

If you find this codebase helpful, please consider citing the SUMO paper.

```bibtex
@article{zhang2026sumo,
  title = {Sumo: Dynamic and Generalizable Whole-Body Loco-Manipulation},
  author = {Zhang, John Z. and Sorokin, Maks and Br{\"u}digam, Jan and Hung, Brandon and Phillips, Stephen and Yershov, Dmitry and Niroui, Farzad and Zhao, Tong and Fermoselle, Leonor and Zhu, Xinghao and Cao, Chao and Ta, Duy and Pang, Tao and Wang, Jiuguang and Culbertson, Preston and Manchester, Zachary and Le Cl\'eac'h, Simon},
  journal = {arXiv preprint arXiv:2604.08508},
  year = {2026},
  url = {https://arxiv.org/abs/2604.08508}
}
```
