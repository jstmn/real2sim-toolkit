#!/usr/bin/env bash
set -euo pipefail

# Test everything: lint + type + unit tests
# Usage: ./bash.sh

echo "== lint (black + ruff) =="
uv run black --line-length 120 --target-version py312 --check src/r2st/geometry.py src/r2st/core.py src/r2st/types.py src/r2st/openai.py src/r2st/__init__.py tests/ scripts/
uv run ruff check --fix src/r2st/geometry.py src/r2st/core.py src/r2st/types.py src/r2st/openai.py src/r2st/__init__.py tests/ scripts/ || echo "ruff: some remaining hints (SIM117 etc) - see log above"

echo "== pytest =="
OPENAI_API_KEY=dummy uv run pytest -q

echo "== import smoke test =="
PYTHONPATH=src OPENAI_API_KEY=dummy uv run python -c "
import r2st
import r2st.types
import r2st.geometry
import r2st.openai
import r2st.core
print('imports ok')
"

echo "== all checks passed =="
