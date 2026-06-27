#!/usr/bin/env bash
# "Big boy model" demo: Qwen3.6-35B-A3B (sparse MoE, ~3B active) on a 24GB card.
# Shows partial-offload (the misconfig ET catches) vs full GPU offload decode tok/s.
#
# NOTE: this model is a SPARSE MoE — decode is already fast (~135 tok/s on a 3090)
# because only ~3B params are active per token. Speculative decoding is KNOWN to be
# net-negative on it, so this script does NOT add a draft model; it demonstrates the
# full-offload win on a big-but-fast model. For a spec-decode WIN, use a DENSE model.
#
# Needs a built llama-server (run prove-on-gpu.sh once, or build llama.cpp).
#   curl -fsSL https://raw.githubusercontent.com/Ronit-Devan/Fixer/main/packages/monitor/validation/prove-35b.sh | bash
set -u
S="${S:-/workspace/llama.cpp/build/bin/llama-server}"
HF="${HF:-unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M}"   # ~21GB; fits a 24GB card tightly
CTX="${CTX:-4096}"                                 # keep KV small so it fits on 24GB
N="${N:-256}"
NGL_PARTIAL="${NGL_PARTIAL:-12}"                   # cripple offload for the baseline
[ -x "$S" ] || { echo "llama-server not found at $S — run prove-on-gpu.sh first (it builds it)"; exit 1; }

wait_up(){ for _ in $(seq 1 360); do curl -fsS localhost:8080/props >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

echo "=== prewarm: downloading Qwen3.6-35B-A3B Q4 (~21GB, one time) ==="
"$S" -hf "$HF" --host 127.0.0.1 --port 8080 --metrics -ngl 999 --ctx-size "$CTX" >/tmp/llama.log 2>&1 &
P=$!
if ! wait_up; then echo "server failed to start (likely VRAM/OOM at -ngl 999 — try a smaller quant). Log:"; tail -25 /tmp/llama.log; kill "$P" 2>/dev/null; exit 1; fi
kill "$P" 2>/dev/null; sleep 3
echo "model ready."

bench(){
  "$S" -hf "$HF" --host 127.0.0.1 --port 8080 --metrics --ctx-size "$CTX" "$@" >/tmp/llama.log 2>&1 &
  local P=$!
  wait_up || { echo "(server didn't come up for: $*; see /tmp/llama.log)" >&2; kill "$P" 2>/dev/null; echo ""; return; }
  curl -fsS localhost:8080/completion \
    -d "{\"prompt\":\"Explain in detail how a mixture-of-experts transformer works.\",\"n_predict\":$N,\"cache_prompt\":false,\"temperature\":0}" 2>/dev/null \
    | python3 -c 'import sys,json;print(json.load(sys.stdin)["timings"]["predicted_per_second"])' 2>/dev/null
  kill "$P" 2>/dev/null; sleep 3
}

echo "=== baseline: -ngl $NGL_PARTIAL (most experts on CPU) ==="
B=$(bench -ngl "$NGL_PARTIAL"); echo "baseline: ${B:-FAILED} tok/s"
echo "=== fixed: -ngl 999 + flash-attn (full GPU offload) ==="
F=$(bench -ngl 999 --flash-attn on); echo "fixed: ${F:-FAILED} tok/s"

echo "=== RESULT (Qwen3.6-35B-A3B Q4, single stream) ==="
if [ -n "$B" ] && [ -n "$F" ]; then
  python3 -c "b=$B; f=$F; print(f'  baseline (partial offload): {b:.1f} tok/s'); print(f'  fixed (full GPU offload)  : {f:.1f} tok/s'); print(f'  >>> {f/b:.2f}x faster'); print('  NOTE: 4090 has ~2.3x the bandwidth of a Blackwell Pro 4000 SFF;'); print('        divide the fixed number by ~2.3 to estimate the SFF.')"
else
  echo "  baseline=${B:-FAILED}  fixed=${F:-FAILED}  — a run failed (likely VRAM); see /tmp/llama.log"
fi
