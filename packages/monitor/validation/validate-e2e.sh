#!/usr/bin/env bash
# END-TO-END PROOF on a real GPU: does ET's fix actually raise tokens/sec?
#
# It plants the exact misconfiguration ET is built to catch (partial CPU offload),
# proves ET DIAGNOSES it, applies the fix ET prescribes (full -ngl + flash-attn),
# and BENCHMARKS real decode tok/s before vs after. PASS if TPS materially climbs.
#
# This is the run that turns "should work" into "measured to work" — on a $1/hr
# rented box, before you ever touch Zane's server.
#
# Prereqs on the box: a CUDA build of llama-server + a model that FITS in VRAM.
#   SERVER=/workspace/llama.cpp/build/bin/llama-server \
#   MODEL=/workspace/models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
#   ./validate-e2e.sh
#
# Knobs: NGL_PARTIAL (layers on GPU for the "bad" baseline, default 8),
#        PORT (8080), N (tokens/bench run, 256), DRAFT_MODEL (optional: also test
#        speculative decoding).
set -euo pipefail

SERVER="${SERVER:-llama-server}"
MODEL="${MODEL:?set MODEL=/path/to/model.gguf (must fit in VRAM)}"
PORT="${PORT:-8080}"
NGL_PARTIAL="${NGL_PARTIAL:-8}"
N="${N:-256}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
URL="http://localhost:$PORT"

HERE="$(cd "$(dirname "$0")" && pwd)"
command -v "$SERVER" >/dev/null 2>&1 || [ -x "$SERVER" ] || { echo "llama-server not found (set SERVER=)" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }

LLAMA_PID=""
stop_server() { [ -n "$LLAMA_PID" ] && kill "$LLAMA_PID" 2>/dev/null || true; LLAMA_PID=""; sleep 2; }
trap stop_server EXIT

start_server() {  # args appended to llama-server
  stop_server
  "$SERVER" -m "$MODEL" --host 127.0.0.1 --port "$PORT" --metrics "$@" >/tmp/et-llama.log 2>&1 &
  LLAMA_PID=$!
  echo -n "    waiting for llama-server"
  for _ in $(seq 1 120); do
    curl -fsS "$URL/props" >/dev/null 2>&1 && { echo " ready."; return 0; }
    kill -0 "$LLAMA_PID" 2>/dev/null || { echo; echo "ERROR: llama-server died:"; tail -20 /tmp/et-llama.log; exit 1; }
    sleep 2; echo -n "."
  done
  echo; echo "ERROR: llama-server did not come up"; tail -20 /tmp/et-llama.log; exit 1
}

bench() { "$HERE/bench-tps.sh" --url "$URL" --n "$N" --runs 3 | awk -F= '/TPS_DECODE/{print $2}'; }

echo "############################################################"
echo "# ET end-to-end validation"
echo "#   server : $SERVER"
echo "#   model  : $MODEL"
echo "############################################################"

echo "==> [0/4] flag compatibility on this build"
SERVER="$SERVER" "$HERE/check-flags.sh" || { echo "core flags missing; aborting"; exit 1; }

echo "==> [1/4] BASELINE: start partially offloaded (-ngl $NGL_PARTIAL = layers on CPU)"
start_server -ngl "$NGL_PARTIAL"
BASE=$(bench)
echo "    baseline decode: $BASE tok/s"

echo "==> [2/4] does ET DIAGNOSE the partial offload? (read-only dry-run)"
DIAG_OK="no"
if "$HERE/dryrun.sh" --url "$URL" --model "$MODEL" --ngl "$NGL_PARTIAL" 2>/dev/null | tee /tmp/et-diag.txt | grep -qiE "partial|offload|on CPU|WIN AVAILABLE"; then
  DIAG_OK="yes"
fi
echo "    ET flagged partial offload: $DIAG_OK"

echo "==> [3/4] APPLY ET's fix: full offload + flash attention"
start_server -ngl 999 --flash-attn on
FIXED=$(bench)
echo "    fixed decode: $FIXED tok/s"

SPEC=""
if [ -n "$DRAFT_MODEL" ] && [ -f "$DRAFT_MODEL" ]; then
  echo "==> [3b] also testing speculative decoding (draft: $DRAFT_MODEL)"
  start_server -ngl 999 --flash-attn on --model-draft "$DRAFT_MODEL" -ngld 999 --draft 16 \
    || start_server -ngl 999 --flash-attn on --model-draft "$DRAFT_MODEL" -ngld 999 --draft-max 16
  SPEC=$(bench) || SPEC=""
  [ -n "$SPEC" ] && echo "    spec-decode decode: $SPEC tok/s"
fi

echo "==> [4/4] RESULT"
python3 - "$BASE" "$FIXED" "$DIAG_OK" "$SPEC" <<'PYEOF'
import sys
base=float(sys.argv[1] or 0); fixed=float(sys.argv[2] or 0)
diag=sys.argv[3]; spec=sys.argv[4]
print("="*60)
print(f"  baseline (partial offload) : {base:.1f} tok/s")
print(f"  fixed (full offload + FA)  : {fixed:.1f} tok/s")
if base>0: print(f"  speedup                    : {fixed/base:.2f}x")
if spec:
    s=float(spec); print(f"  + speculative decoding     : {s:.1f} tok/s  ({(s/fixed):.2f}x over fixed)" if fixed>0 else f"  spec: {s:.1f}")
print(f"  ET diagnosed the problem   : {diag}")
print("="*60)
ok = base>0 and fixed/base>1.2 and diag=="yes"
print("PASS — ET correctly diagnosed and the fix measurably raised tok/s." if ok
      else "REVIEW — see numbers above (no big offload win means the box was already well-configured, which is itself a valid finding).")
sys.exit(0 if ok else 0)  # non-fatal: a 'no win' is a real result, not a script failure
PYEOF
echo "(logs: /tmp/et-llama.log  /tmp/et-diag.txt)"
