#!/usr/bin/env bash
# One command to run the ET monitor on this machine (macOS / Linux).
#   ./run.sh                      auto-detect GPU, find llama-server on :8080
#   ./run.sh --gpu-price 0.50     show $ wasted to idle
#   ./run.sh --demo               scripted timeline, no GPU/model needed
#
# Override the interpreter on a box where python3 isn't the right one:
#   PYTHON=/usr/bin/python3.11 ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Error: '$PY' not found on PATH. Install Python 3.10+, or set PYTHON=/path/to/python." >&2
  exit 1
fi

# Reuse an existing venv only if its interpreter is actually present AND is
# Python 3.10+. Probing the interpreter (not just the .venv directory) recovers
# from a half-created or stale venv left by an interrupted first run — which
# otherwise wedges every later launch with a confusing "no such file" error. A
# version-mismatched venv (built by an older Python) is rebuilt here instead of
# failing later inside pip.
VENV_PY=".venv/bin/python"
need_venv=1
if [ -x "$VENV_PY" ] \
  && "$VENV_PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
  need_venv=0
fi
if [ "$need_venv" -eq 1 ]; then
  if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    ver="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo '?')"
    echo "Error: Python 3.10+ is required, but '$PY' is $ver." >&2
    echo "  Install a newer Python, or set PYTHON=/path/to/python3.11 and re-run." >&2
    exit 1
  fi
  rm -rf .venv
  echo "Creating virtual environment..."
  if ! "$PY" -m venv .venv; then
    echo "Error: failed to create the virtualenv with '$PY'." >&2
    echo "  On Debian/Ubuntu, install the venv module:  sudo apt-get install -y python3-venv" >&2
    exit 1
  fi
fi

"$VENV_PY" -m pip install -q --upgrade pip
# Try with the NVIDIA telemetry extra; fall back to core if the wheel won't build
# (the app then uses nvidia-smi, and finally mock data, automatically).
if ! "$VENV_PY" -m pip install -q -e ".[gpu]" 2>/dev/null; then
  echo "nvidia-ml-py unavailable; installing core only (nvidia-smi / mock fallback)."
  "$VENV_PY" -m pip install -q -e .
fi

exec ./.venv/bin/et-monitor "$@"
