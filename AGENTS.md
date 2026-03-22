# Repository Guidelines

## Project Structure & Module Organization
`sumo/` is the active package. Put task code in `sumo/tasks/g1/` or `sumo/tasks/spot/`, controller wrappers in `sumo/controller/`, CLI and MPC entrypoints in `sumo/cli.py` and `sumo/run_mpc/`, and MuJoCo XML/mesh assets under `sumo/models/`. `g1_extensions/` contains the optional pybind11 extension for G1 backends. `tests/` holds the pytest suite. Treat `judo-private/` as an old fork used only as migration reference; match new code to the public `judo-rai` repo instead. Use `.judo-src/` as the local reference checkout when porting behavior.

## Build, Test, and Development Commands
Use `pixi` by default:

```bash
pixi install
pixi run build
pixi run build-judo-ext
pixi run pytest tests/ -v
pixi run sumo --init-task g1_box --num-episodes 1
```

`pixi run build` compiles `g1_extensions`; `pixi run build-judo-ext` builds Judo's `mujoco_extensions` backend for Spot tasks. `pixi install -e dev` remains equivalent to `pixi install` for this repo, but `pixi run` without `-e` already targets the default full app environment. Use `pip install -e .` only as a lightweight fallback when you do not need the managed `pixi` environment.

## Coding Style & Naming Conventions
Write Python with 4-space indentation, explicit imports, and type-aware dataclass configs. Follow existing naming: `snake_case` for modules/functions, `PascalCase` for classes, and `*Config` for task/config dataclasses such as `G1BoxConfig`. New task modules should follow patterns like `g1_box.py` or `spot_table_drag.py`. Prefer `ruff check .` before submitting changes. When porting code, preserve `judo-rai` APIs and conventions first, even if `judo-private/` differs.

## Testing Guidelines
Add tests in `tests/` and name files `test_*.py`. Cover imports, task registration, and task-specific behavior such as `reset()`, `reward()`, and control shape. Mark extension-only tests with `@pytest.mark.g1_extensions`; `tests/conftest.py` skips them when `g1_extensions` is unavailable. When adding a task, update registration in `sumo/tasks/__init__.py` and add at least one smoke test.

## Commit & Pull Request Guidelines
The repo history is minimal, so use short imperative commit subjects. Scoped messages are preferred, for example `feat(tasks): port spot_table_drag`. PRs should summarize the port or feature, note what was aligned to public `judo-rai`, list commands run, and include screenshots or logs for simulator-facing changes. Avoid mixing migration cleanup with unrelated refactors.
