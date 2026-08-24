#!/usr/bin/env bash
set -euo pipefail

# Format with Black (120-char line length) and lint with ruff.
# Skip the vendored SAM 3 tree.
uv run black --line-length 120 --target-version py312 --extend-exclude '/sam3/' .
uv run ruff check --fix --exclude src/r2st/sam3 .
