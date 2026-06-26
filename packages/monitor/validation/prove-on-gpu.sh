#!/usr/bin/env bash
# Self-contained on-GPU proof: build llama.cpp (CUDA), then benchmark real
# decode tok/s with the model PARTIALLY on CPU (the misconfiguration ET catches)
# vs FULLY offloaded (the fix ET applies). Prints the before/after speedup.
#
# Run on a fresh RunPod CUDA/PyTorch pod with ONE line (no copy-paste of a big
# block, nothing to mangle):
#   curl -fsSL https://raw.githubusercontent.com/Ronit-Devan/Fixer/main/packages/monitor/validation/prove-on-gpu.sh | bash
#
# Override anything via env: HF=<repo:quant>  NGL_PARTIAL=8  N=256
set -e
export PATH="/usr/local/cuda/bin:$PATH"
HF="${HF:-bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M}"
NGL_PARTIAL="${NGL_PARTIAL:-8}"
N="${N:-256}"

echo "=== [1/3] deps + build llama.cpp (CUDA) - a few minutes ==="
apt-get update -y -qq && apt-get install -y -qq build-essential cmake git libcurl4-openssl-dev curl python3
command -v nvidia-smi >/dev/null || { echo "NO GPU ON THIS POD"; exit 1; }
command -v nvcc >/dev/null || { echo "NO CUDA COMPILER - use the RunPod PyTorch template"; exit 1; }
mkdir -p /workspace && cd /workspace
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
cmake -B build -DGGML_CUDA=ON ${ARCH:+-DCMAKE_CUDA_ARCHITECTURES=$ARCH} >/dev/null
cmake --build build --config Release -j"$(nproc)" --target llama-server
S=/workspace/llama.cpp/build/bin/llama-server
[ -x "$S" ] || { echo "BUILD FAILED"; exit 1; }
echo "=== BUILD OK ==="

echo "=== [2/3] benchmark (model downloads once, ~2GB) ==="
wait_up(){ for _ in $(seq 1 300); do curl -fsS localhost:8080/props >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

# Pre-warm: download + cache the model BEFORE timing anything, so the baseline
# run never races the download (that left the first baseline with no reading).
echo "-- prewarm: caching the model (one time) --"
"$S" -hf "$HF" --host 127.0.0.1 --port 8080 --metrics -ngl 999 >/tmp/llama.log 2>&1 &
PW=$!
if ! wait_up; then echo "server failed to start; last log lines:"; tail -25 /tmp/llama.log; kill "$PW" 2>/dev/null; exit 1; fi
kill "$PW" 2>/dev/null; sleep 3
echo "   model ready."

bench(){
  "$S" -hf "$HF" --host 127.0.0.1 --port 8080 --metrics "$@" >/tmp/llama.log 2>&1 &
  local P=$!
  wait_up || { kill "$P" 2>/dev/null; echo ""; return; }
  curl -fsS localhost:8080/completion \
    -d "{\"prompt\":\"Explain in detail how GPUs work.\",\"n_predict\":$N,\"cache_prompt\":false,\"temperature\":0}" 2>/dev/null \
    | python3 -c 'import sys,json;print(json.load(sys.stdin)["timings"]["predicted_per_second"])' 2>/dev/null
  kill "$P" 2>/dev/null; sleep 3
}
echo "-- baseline: partial offload (-ngl $NGL_PARTIAL, layers stuck on CPU) --"
B=$(bench -ngl "$NGL_PARTIAL"); echo "baseline: ${B:-FAILED} tok/s"
echo "-- fixed: full GPU offload (-ngl 999) --"
F=$(bench -ngl 999); echo "fixed: ${F:-FAILED} tok/s"

echo "=== [3/3] RESULT ==="
if [ -n "$B" ] && [ -n "$F" ]; then
  python3 -c "b=$B; f=$F; print(f'  baseline (partial offload): {b:.1f} tok/s'); print(f'  fixed (full GPU offload)  : {f:.1f} tok/s'); print(f'  >>> {f/b:.2f}x faster  (this is the optimization ET applies)')"
else
  echo "  baseline=${B:-FAILED}  fixed=${F:-FAILED}  — a run failed; see /tmp/llama.log"
fi
