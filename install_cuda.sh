#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pip install -U pip
if ! python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu128; then
  echo "cu128 wheel failed, trying cu126..."
  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu126
fi
python3 -m pip install -r requirements.txt
python3 -m pip install datasets
python3 scripts/check_cuda.py
