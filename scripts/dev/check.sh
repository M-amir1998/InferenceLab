#!/usr/bin/env bash
# Local quality gates, identical to the Jenkins pipeline stages (no GPU needed).
set -euo pipefail
uv lock --check
uv sync --frozen --all-extras
uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen mypy
uv run --frozen pytest -q
bash scripts/ci/secret_scan.sh
uv run --frozen bandit -c pyproject.toml -r src -q
uv build
