# Repository Guidelines

## Project Structure & Module Organization
`sumo/` is the main Python package. Task implementations live in `sumo/tasks/g1/` and `sumo/tasks/spot/`; shared controller code is in `sumo/controller/`; CLI and headless MPC entry points are `sumo/cli.py` and `sumo/run_mpc/`. MuJoCo XML and mesh assets are packaged under `sumo/models/`. Native G1 rollout bindings live in `g1_extensions/` with CMake/pybind11 sources. Tests belong in `tests/`, while generated run artifacts should stay in `out/`, `outputs/`, or `run_mpc/results/`.

## Build, Test, and Development Commands
Use Pixi as the default environment manager.

```bash
pixi install                         # create the default dev environment
pixi run build                       # build g1_extensions and Judo MuJoCo extensions
pixi run sumo task=spot_box_push     # launch the interactive app
pixi run python -m sumo.run_mpc --init-task=g1_box --num-episodes=2
pixi run pytest tests/ -v            # run the test suite
pixi run pre-commit run --all-files  # run Ruff, formatting, and whitespace hooks
pixi run pyright sumo/               # run static type checks
```

Run `pixi run build` before G1 or Spot simulations that require compiled extension backends.

## Coding Style & Naming Conventions
Python uses 4-space indentation, explicit imports, and Ruff formatting with a 120-character line length. Keep module and function names in `snake_case`, classes in `PascalCase`, and task configs named with a `Config` suffix, for example `G1BoxConfig`. New task files should follow existing patterns such as `g1_table_push.py` or `spot_tire_roll.py`. Prefer dataclass-style configs and match public `judo-rai` APIs rather than introducing local compatibility shims.

## Testing Guidelines
Use pytest. Name files `test_*.py` and tests `test_*`. Add coverage for imports, task registration, reset behavior, reward shape, and control dimensions when adding or changing tasks. Mark tests that require the optional native G1 extension with `@pytest.mark.g1_extensions`; `tests/conftest.py` skips those when the extension is not built. Run `pixi run pytest -rsx` to mirror CI output.

## Commit & Pull Request Guidelines
The existing history uses short, imperative subjects such as `Update README.md` and `clean up paper tasks to match paper rewards`. Keep commits focused and avoid mixing task behavior changes with cleanup. Before pushing or requesting review, run `pixi run pre-commit run --all-files` and `pixi run pyright sumo/` locally. PRs should include a concise summary, commands run, and screenshots or logs for simulator-facing changes.

## Security & Configuration Tips
Do not commit generated build directories, local `.judo-src/` checkouts, cache folders, or large simulation outputs. Keep dependency and build changes in `pyproject.toml`, `pixi.lock`, and the relevant CMake files together so CI can reproduce them.
