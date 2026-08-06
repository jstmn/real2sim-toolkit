#!/usr/bin/env bash
set -euo pipefail

# Format with Black (120-char line length) and lint with ruff.
uv run black --line-length 120 --target-version py312 .
uv run ruff check --fix .
