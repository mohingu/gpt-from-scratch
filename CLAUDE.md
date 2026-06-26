# gpt-from-scratch

A small decoder-only transformer built from scratch in PyTorch, following Karpathy's
*Neural Networks: Zero to Hero*. The goal is understanding: every component (autograd, the
language-modelling objective, self-attention) is built by hand before relying on the
framework version.

## Layout

- `components/` - each build component as a self-contained package, landing in order:
  - `components/micrograd/` - scalar autograd engine (`engine.py`) and a tiny neural-net
    library (`nn.py`). **Done.**
  - `components/bigram/` - character-level bigram language model (makemore pt 1). *Next.*
  - `components/gpt/` - decoder-only transformer trained on tiny Shakespeare. *Main event.*
- `notebooks/` - code-along scratchpads; the `.py` modules are refactored out of these.
- `tests/` - pytest suite.
- `data/` - training data and checkpoints (gitignored).

## Conventions

- Python 3.12+. Type hints on all functions (including `-> None`).
- Imports *within* a component are relative (`from .engine import Value`). Cross-component
  and test imports use the full path (`from components.micrograd.nn import MLP`).
- The project is not pip-installed (no build backend). pytest puts the repo root on the path
  via `pythonpath = ["."]` in `pyproject.toml`, so `import components...` resolves.

## Environment and tooling

- `uv` manages the env and dependencies. `make env` creates the venv and installs the
  pre-commit and nbstripout hooks.
- `ruff` handles lint and formatting (run on commit via pre-commit). Notebooks are excluded
  from linting; nbstripout strips their outputs on commit.
- Supply-chain cooldown: `[tool.uv] exclude-newer` in `pyproject.toml` pins resolution to
  packages published before a fixed date. Bump it deliberately to pull in newer releases.
- After changing dependencies in `pyproject.toml`, run `uv lock` and commit `uv.lock` - CI
  runs `uv sync --locked` and fails if the lockfile is out of sync.

## Testing

- `uv run pytest` (or `make check` to run all pre-commit hooks too).
- The micrograd engine is validated by gradient-checking against PyTorch's autograd: the same
  expression is replayed through both and forward values plus backward gradients must agree.

## CI

- GitHub Actions (`.github/workflows/ci.yml`): `uv sync --locked`, `ruff check`,
  `ruff format --check`, then `pytest`. Runs on pushes to `main` and on pull requests.

## Workflow

- Branch off `main`, open a PR, and let CI pass before merging.

## Make targets

`make env` (set up), `make lint` (ruff --fix), `make format`, `make check` (all hooks),
`make lab` (Jupyter Lab), `make clean` (remove caches).
