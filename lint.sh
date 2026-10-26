#!/usr/bin/env bash
set -euo pipefail

# Format with Black (120-char line length) and lint with ruff.
# Skip the vendored SAM 3 and S2M2 trees.
uv run black --line-length 120 --target-version py312 --extend-exclude '/(sam3|S2M2)/' .
uv run ruff check --fix --exclude src/r2st/sam3 --exclude src/r2st/S2M2 .
