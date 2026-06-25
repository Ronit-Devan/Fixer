#!/usr/bin/env bash
# One-shot ET test setup on a fresh CUDA GPU box (Ubuntu, e.g. a RunPod pod).
#
# Builds llama.cpp with CUDA, starts llama-server with a small model and
# metrics enabled, clones ET, and launches the monitor. Both run in tmux so
# they survive you closing the terminal.
#
# Usage (on the pod's terminal):
#   curl -fsSL https://raw.githubusercontent.com/Ronit-Devan/Fixer/main/packages/monitor/setup-on-gpu.sh | GPU_PRICE=0.69 bash
#
# Override anything via env vars:
#   MODEL=bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M  GPU_PRICE=0.69  MON_PORT=7070  bash setup-on-gpu.sh
set -euo pipefail

MODEL="${MODEL:-bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M}"
GPU_PRICE="${GPU_PRICE:-0.69}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
MON_PORT="${MON_PORT:-7070}"
WORK="${WORK:-/workspace}"
JOBS="${JOBS:-$(nproc)}"   # use all cores

mkdir -p "$WORK"; cd "$WORK"

echo "==> [1/4] installing build deps"
if command -v apt-get >/dev/null; then
  apt-get update -y -qq
  apt-get install -y -qq build-essential cmake git libcurl4-openssl-dev tmux curl
fi

echo "==> [2/4] building llama.cpp with CUDA (a few minutes)"
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
# Build ONLY for this GPU's compute capability and ONLY the server binary.
# The default build compiles flash-attention kernels for every architecture
# (hundreds of extra .cu files) and all the example/test binaries -- ~3x slower
# for no benefit on a single known card.
ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
CUDA_ARCH_FLAG=""
[ -n "$ARCH" ] && CUDA_ARCH_FLAG="-DCMAKE_CUDA_ARCHITECTURES=$ARCH" && echo "    targeting CUDA arch $ARCH only"
cmake -B build -DGGML_CUDA=ON $CUDA_ARCH_FLAG >/dev/null
cmake --build build --config Release -j "$JOBS" --target llama-server
SERVER="$WORK/llama.cpp/build/bin/llama-server"
cd "$WORK"

echo "==> [3/4] starting llama-server (model downloads on first run)"
tmux kill-session -t llama 2>/dev/null || true
tmux new-session -d -s llama \
  "$SERVER -hf $MODEL --host 0.0.0.0 --port $LLAMA_PORT --metrics -ngl 999 2>&1 | tee $WORK/llama.log"

echo "==> [4/4] starting the ET monitor"
[ -d Fixer ] || git clone --depth 1 https://github.com/Ronit-Devan/Fixer
MON_DIR="$WORK/Fixer/packages/monitor"

# Wait for llama-server to finish loading the model (download can take minutes),
# then capture the decode roofline (model size + layers + GPU bandwidth) so the
# monitor has MBU / single-stream-ceiling / partial-offload from the first tick.
echo "    waiting for llama-server, then detecting the decode roofline..."
for _ in $(seq 1 90); do
  curl -fsS "http://localhost:$LLAMA_PORT/props" >/dev/null 2>&1 && break
  sleep 5
done
( cd "$MON_DIR" && ./run.sh --detect --llama-url "http://localhost:$LLAMA_PORT" 2>&1 | tee -a "$WORK/monitor.log" ) || true

tmux kill-session -t monitor 2>/dev/null || true
tmux new-session -d -s monitor \
  "cd $MON_DIR && ./run.sh --gpu-price $GPU_PRICE --llama-url http://localhost:$LLAMA_PORT --host 0.0.0.0 --port $MON_PORT --no-browser 2>&1 | tee $WORK/monitor.log"

cat <<EOF

==================================================================
  ET is starting up.

  Dashboard:  open the pod's HTTP port $MON_PORT
              (RunPod proxy URL: https://<POD_ID>-$MON_PORT.proxy.runpod.net)
  Report:     same URL + /report

  Logs / attach:
    tmux attach -t llama      # llama-server (Ctrl-b d to detach)
    tmux attach -t monitor    # ET monitor
    tail -f $WORK/llama.log $WORK/monitor.log

  The model downloads on first run; give llama-server a few minutes.

  Generate load to move the verdicts:
    curl http://localhost:$LLAMA_PORT/completion \\
      -d '{"prompt":"Write a 500-word essay on GPUs","n_predict":500}'
==================================================================
EOF
