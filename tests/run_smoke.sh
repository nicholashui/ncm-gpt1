#!/usr/bin/env bash
# CPU smoke: unit tests (including a short end-to-end train + generate).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
export PYTHONPATH="$(pwd)"
export PYTHONUNBUFFERED=1
echo "Using $PYTHON"
"$PYTHON" tests/test_ncm_gpt1.py
echo "SMOKE OK"
