#!/usr/bin/env bash
# ZERO-RISK diagnosis against a live llama-server (e.g. Zane's box).
#
# Reads ONLY /props and /metrics (HTTP GET) — never restarts, never sends a
# write, never touches the server. Runs the real ET analyzer for a few seconds
# and prints the verdict + the EXACT command it WOULD run, so you learn whether
# there is a real TPS win on the table before spending a dollar or risking prod.
#
# Usage:
#   ./dryrun.sh --url http://ZANE_HOST:8080
#   ./dryrun.sh --url http://host:8080 --model /models/x.gguf --ngl 999 --gpu-bandwidth 3350
set -euo pipefail

URL="http://localhost:8080"
PORT="${PORT:-7099}"
SECONDS_RUN="${SECONDS_RUN:-8}"
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --model) EXTRA+=(--model "$2"); shift 2;;
    --ngl) EXTRA+=(--ngl "$2"); shift 2;;
    --gpu-bandwidth) EXTRA+=(--gpu-bandwidth "$2"); shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
MON_DIR="$(dirname "$HERE")"            # packages/monitor
PY="${PYTHON:-python3}"
cd "$MON_DIR"

# venv (reuse run.sh's; create if missing).
if [ ! -x ./.venv/bin/et-monitor ]; then
  echo "==> setting up monitor venv (one-time)"
  "$PY" -m venv .venv
  ./.venv/bin/pip install -q --upgrade pip
  ./.venv/bin/pip install -q -e ".[gpu]" 2>/dev/null || ./.venv/bin/pip install -q -e .
fi
BIN=./.venv/bin/et-monitor

echo "==> [1/2] detecting decode roofline from $URL (read-only)"
$BIN --detect --llama-url "$URL" "${EXTRA[@]}" || echo "    (detect best-effort; continuing)"

echo "==> [2/2] running the analyzer in ADVISE mode for ${SECONDS_RUN}s (touches nothing)"
$BIN --llama-url "$URL" --remediation-mode advise --port "$PORT" \
     --no-browser --interval 1 "${EXTRA[@]}" >/tmp/et-dryrun.log 2>&1 &
MON_PID=$!
cleanup() { kill "$MON_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Poll the local analyzer until it has a real verdict (needs a few samples).
DIAG=""
for _ in $(seq 1 "$((SECONDS_RUN + 6))"); do
  sleep 1
  DIAG="$(curl -fsS "http://127.0.0.1:$PORT/api/diagnosis" 2>/dev/null || true)"
  v="$(printf '%s' "$DIAG" | $PY -c 'import json,sys;d=json.load(sys.stdin);print(d.get("verdict",""))' 2>/dev/null || true)"
  [ -n "$v" ] && [ "$v" != "unknown" ] && break
done

[ -n "$DIAG" ] || { echo "ERROR: could not reach the local analyzer; see /tmp/et-dryrun.log" >&2; cat /tmp/et-dryrun.log >&2; exit 1; }

printf '%s' "$DIAG" | $PY - <<'PYEOF'
import json,sys
d=json.load(sys.stdin); m=d.get("metrics",{}) or {}
def g(k,f="—"):
    v=m.get(k); return v if v is not None else f
print("="*64)
print(f"VERDICT : {d.get('verdict','?')}  ({d.get('severity','?')}, confidence {d.get('confidence','?')})")
print(f"SUMMARY : {d.get('summary','')}")
print("-"*64)
print(f"  decode tok/s        : {g('gen_tokens_per_s')}")
print(f"  single-stream ceiling: {g('ceiling_tok_s')} tok/s")
print(f"  MBU (bandwidth use) : {g('mbu')}")
print(f"  offload fraction    : {g('offload_fraction')}   (1.0 = whole model on GPU)")
print(f"  throughput vs ceiling: {g('throughput_pct')}")
print(f"  mean GPU util       : {g('mean_util_pct')}%")
print(f"  VRAM used           : {g('mem_used_ratio')}")
print("-"*64)
recs=d.get("recommendations") or []
if recs:
    print("WHAT ET WOULD DO / ADVISE:")
    for r in recs: print(f"  • {r}")
# The money read:
off=m.get("offload_fraction"); wall=m.get("at_practical_ceiling")
print("="*64)
if off is not None and off < 0.98:
    print(">> WIN AVAILABLE: model is partially on CPU. Full offload (-ngl 999) is the big lever.")
elif wall:
    print(">> AT THE WALL: single-stream is bandwidth-bound. Only spec-decode / smaller quant / batching move it.")
else:
    print(">> Looks well-configured on offload. Read MBU/ceiling above to judge remaining headroom.")
PYEOF

echo
echo "(advise-only; the target server was never modified. Full log: /tmp/et-dryrun.log)"
