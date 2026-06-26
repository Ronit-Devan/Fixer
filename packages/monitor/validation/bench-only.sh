#!/usr/bin/env bash
# Benchmark only (no build). Use after prove-on-gpu.sh once llama.cpp is built
# and the model is cached. Pre-warms the model FIRST so the timed baseline never
# races a download, then measures partial-offload vs full-offload decode tok/s.
#   curl -fsSL https://raw.githubusercontent.com/Ronit-Devan/Fixer/main/packages/monitor/validation/bench-only.sh | bash
set -u
S="${S:-/workspace/llama.cpp/build/bin/llama-server}"
HF="${HF:-bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M}"
N="${N:-256}"
NGL_PARTIAL="${NGL_PARTIAL:-8}"
[ -x "$S" ] || { echo "llama-server not found at $S — run prove-on-gpu.sh first"; exit 1; }

wait_up(){ for _ in $(seq 1 300); do curl -fsS localhost:8080/props >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

echo "=== prewarm: downloading/caching the model (one time) ==="
"$S" -hf "$HF" --host 127.0.0.1 --port 8080 --metrics -ngl 999 >/tmp/llama.log 2>&1 &
P=$!
if ! wait_up; then echo "server failed to start; last log lines:"; tail -25 /tmp/llama.log; kill "$P" 2>/dev/null; exit 1; fi
kill "$P" 2>/dev/null; sleep 3
echo "model ready."

bench(){
  "$S" -hf "$HF" --host 127.0.0.1 --port 8080 --metrics "$@" >/tmp/llama.log 2>&1 &
  local P=$!
  wait_up || { echo "(server didn't come up for: $*)" >&2; kill "$P" 2>/dev/null; echo ""; return; }
  curl -fsS localhost:8080/completion \
    -d "{\"prompt\":\"Explain in detail how GPUs work.\",\"n_predict\":$N,\"cache_prompt\":false,\"temperature\":0}" 2>/dev/null \
    | python3 -c 'import sys,json;print(json.load(sys.stdin)["timings"]["predicted_per_second"])' 2>/dev/null
  kill "$P" 2>/dev/null; sleep 3
}

echo "=== baseline: -ngl $NGL_PARTIAL (partial offload, layers on CPU) ==="
B=$(bench -ngl "$NGL_PARTIAL"); echo "baseline: ${B:-FAILED} tok/s"
echo "=== fixed: -ngl 999 (full GPU offload) ==="
F=$(bench -ngl 999); echo "fixed: ${F:-FAILED} tok/s"

echo "=== RESULT ==="
if [ -n "$B" ] && [ -n "$F" ]; then
  python3 -c "b=$B; f=$F; print(f'  baseline (partial offload): {b:.1f} tok/s'); print(f'  fixed (full GPU offload)  : {f:.1f} tok/s'); print(f'  >>> {f/b:.2f}x faster  (the optimization ET applies)')"
else
  echo "  baseline=${B:-FAILED}  fixed=${F:-FAILED}  — a run failed; see /tmp/llama.log"
fi
