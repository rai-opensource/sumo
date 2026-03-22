# Sumo 🥋

`sumo`🥋 is a research codebase for whole-body loco-manipulation built on
[`judo`🥋](https://github.com/bdaiinstitute/judo).

## Install

Requires [pixi](https://pixi.sh).

```bash
# Install the default app environment
pixi install

# Build the local g1_extensions extension (required for G1 tasks)
pixi run build

# Build judo's mujoco_extensions rollout backend (required for Spot tasks)
pixi run build-judo-ext
```

`pixi run sumo` assumes both native extensions above have already been built.

If you prefer named environments, `pixi install -e dev` is equivalent for this repo,
and the corresponding task form is `pixi run -e dev ...`.

## Run

```bash
# Launch simulation + visualizer with default task
pixi run sumo

# Specify a task
pixi run sumo --init-task g1_box --num-episodes 1 --episode-length-s 1
pixi run sumo --init-task spot_box_push --num-episodes 1
```

## Tests

```bash
pixi run pytest tests/ -v
```

## Reference

If you find this codebase helpful, please consider citing the SUMO paper.

```bibtex
@article{zhang2026sumo,
  title={Sumo: Dynamic and Generalizable Whole-Body Loco-Manipulation},
  author={John Z. Zhang and Maks Sorokin and Jan Brudigam and Brandon Hung and Stephen Phillips and Dmitry Yershov and Farzad Niroui and Tong Zhao and Leonor Fermoselle and Xinghao Zhu and Chao Cao and Duy Ta and Tao Pang and Jiuguang Wang and Preston Culbertson and Zachary Manchester and Simon Le Cleac'h},
  year={2026},
  note={TODO: add arXiv link}
}
```
