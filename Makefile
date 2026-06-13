.PHONY: env lint format check lab clean

# Create the uv environment, sync all deps (incl. dev), and install pre-commit hooks
env:
	uv sync
	uv run pre-commit install
	uv run nbstripout --install

# Lint and auto-fix with ruff
lint:
	uv run ruff check --fix .

# Format with ruff
format:
	uv run ruff format .

# Run all pre-commit hooks across the repo
check:
	uv run pre-commit run --all-files

# Launch Jupyter Lab for the code-along
lab:
	uv run jupyter lab

# Remove caches
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf .ruff_cache
