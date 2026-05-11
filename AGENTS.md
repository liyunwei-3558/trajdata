# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python package for trajectory dataset loading and analysis. Core source lives in `src/trajdata/`, with dataset adapters under `src/trajdata/dataset_specific/`, map code under `src/trajdata/maps/`, caching under `src/trajdata/caching/`, and batch/state structures under `src/trajdata/data_structures/`. Tests are in `tests/`. SinD-specific docs and workflows are in `Task_docs/`, `SIND_*.md`, `DATASETS.md`, `scripts/`, and `SinD_integration_test_scripts/`. Risk mining utilities and reports live in `risk_mining/`. The SinD simulation runner is isolated under `Simulation_test_toolchain/`, and semantic scenario label tooling is under `semantic_labels/`.

## Build, Test, and Development Commands

Install dependencies and the editable package from the repository root:

```bash
pip install -r requirements.txt
pip install -e .
```

Useful development commands:

```bash
pytest tests/                 # Run the unit test suite
pytest -n auto tests/         # Run tests in parallel
python -m unittest tests/test_state.py
black src/trajdata risk_mining
isort src/trajdata risk_mining
python -m build               # Build source/wheel distributions
```

Run SinD integration checks individually, for example `python SinD_integration_test_scripts/test6_lanelet2_map.py`, because many require local dataset paths and optional Lanelet2 support.

Run the SinD simulation toolchain with the `trajdata` conda environment:

```bash
conda run -n trajdata python -m Simulation_test_toolchain.run_test \
  --config Simulation_test_toolchain/test_projects/example_semantic_label/config.yaml
```

Regenerate semantic labels with `python -m semantic_labels.import_sind_semantic_labels`.

## Coding Style & Naming Conventions

Target Python 3.8+ for the package metadata, while the local environment file assumes Python 3.10+. Use 4-space indentation, type hints where they clarify public interfaces, and Black formatting. Imports should be sorted with isort using the Black profile configured in `pyproject.toml`. Use `snake_case` for functions, modules, and variables; `PascalCase` for classes and dataclasses. Dataset adapters should follow existing names such as `nusc_dataset.py`, `sind_utils.py`, and `RawDataset` subclasses.

## Testing Guidelines

Add focused pytest tests in `tests/test_*.py` for package behavior. Prefer small synthetic inputs for data structures, map utilities, and filtering logic. For dataset-specific changes, include cache/loading tests when feasible and document any raw-data prerequisite. SinD visual or integration scripts are supplementary checks, not replacements for unit tests.

## Commit & Pull Request Guidelines

Recent commits use short imperative messages, for example `Add SinD signal-aware risk metrics` and `Improve batch extraction monitoring and caching`. Keep commits scoped to one logical change. Pull requests should describe the behavior change, list tests run, mention dataset or cache assumptions, and include screenshots or generated artifact paths for visualization changes.

## Data & Configuration Notes

Do not commit raw datasets, generated caches, or large local outputs. Default caches are expected under `~/.unified_data_cache/`. Keep dataset paths configurable through examples or docs, and avoid hard-coding machine-specific absolute paths in library code.
