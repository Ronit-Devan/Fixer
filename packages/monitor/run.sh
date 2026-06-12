#!/usr/bin/env bash
# One command to run the ET monitor on this machine (macOS / Linux).
#   ./run.sh                      auto-detect GPU, find llama-server on :8080
#   ./run.sh --gpu-price 0.50     show $ wasted to idle
#   ./run.sh --demo               scripted timeline, no GPU/model needed
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  echo "Creating virtual environment..."
  "$PY" -m venv .venv
fi

./.venv/bin/python -m pip install -q --upgrade pip
# Try with the NVIDIA telemetry extra; fall back to core if the wheel won't build.
if ! ./.venv/bin/pip install -q -e ".[gpu]" 2>/dev/null; then
  echo "nvidia-ml-py unavailable; installing core only (nvidia-smi / mock fallback)."
  ./.venv/bin/pip install -q -e .
fi

exec ./.venv/bin/et-monitor "$@"
