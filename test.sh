#!/usr/bin/env bash
set -euo pipefail

uv run pytest -s tests/ --capture=no --disable-warnings
