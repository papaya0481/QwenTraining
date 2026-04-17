# Repository Guidelines

## Project Structure & Module Organization
This repository combines several related codebases:

- `verl/`: primary RL training engine and the main place for PPO/DAPO/FSDP changes.
- `scripts/`: local entrypoints and task-specific runners, especially `scripts/dapo/`.
- `tests/`: repo-level utility and regression scripts such as `tests/compare_safetensors.py`.
- `ms-swift/`: upstream fine-tuning toolkit used for SFT-style workflows.
- `LiveCodeBench/`: evaluation harness and assets.
- `data/`, `docs/`: datasets, notes, and project documentation.

When changing training behavior, check both `scripts/dapo/config/` and the corresponding implementation under `verl/verl/`.

## Build, Test, and Development Commands
Run commands from the repo root unless a subproject path is shown.

- `bash scripts/dapo/run_dapo.sh`: run the local DAPO training pipeline.
- `python3 -m scripts.dapo.main_dapo --config-path=scripts/dapo/config --config-name=dapo_qwen3_5_0_8b`: launch DAPO directly.
- `python3 -m pytest verl/tests`: run VERL tests if `pytest` is installed.
- `python3 -m py_compile <files...>`: fast syntax check for touched Python files.
- `cd verl && pre-commit run -a`: run Ruff, mypy, config generation, and sanity checks.
- `cd ms-swift && make test` or `make linter`: run Swift CI helpers.

## Coding Style & Naming Conventions
- Use 4-space indentation and keep Python code ASCII unless the file already uses Unicode.
- Follow Ruff settings in `verl/pyproject.toml`; line length is `120`.
- Prefer descriptive snake_case for functions, variables, config keys, and test names.
- Keep YAML config names task-specific, e.g. `dapo_qwen3_5_0_8b.yaml`.
- Use `verl`, `SGLang`, and `sglang` casing consistently; the pre-commit hooks enforce this.

## Testing Guidelines
- Main test framework: `pytest` under `verl/tests/` and `ms-swift/tests/`.
- Name test files `test_*.py`.
- Add focused regression tests for save/load, dtype, checkpoint, and config behavior when touching training internals.
- For infra-limited environments, at minimum run `py_compile` on edited files and note any skipped tests.

## Commit & Pull Request Guidelines
- Recent history uses short, imperative commit messages, e.g. `add verl saving bf16 support`.
- Keep commits scoped to one logical change.
- PRs should include:
- a short problem statement,
- the files or subsystems changed,
- exact verification commands run,
- any config or checkpoint compatibility impact.

## Configuration & Safety Notes
- Prefer editing VERL save/load behavior in code over hard-coding paths in YAML.
- Do not assume model weights or checkpoints exist locally; make path handling robust.
- For LoRA changes, verify both save layout and dtype behavior.
