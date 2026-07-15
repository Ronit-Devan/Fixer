#!/usr/bin/env bash
# One-shot client replica test: validate ET on the client's MODEL CLASS + BANDWIDTH
# without the client's box. Runs a real MoE on real llama-server and points ET at
# it with the client's SFF bandwidth (--gpu-bandwidth 432), so ET's GGUF read,
# MoE roofline, prefix-cache observability, and prefill diagnosis run on real data.
#
#   curl -fsSL https://raw.githubusercontent.com/Ronit-Devan/Fixer/main/packages/monitor/validation/test-client-replica.sh | bash
#
# RENT A 48 GB GPU (L40S / A6000, ~$1/hr): the client runs 30B-A3B at 30K ctx on
# 24 GB, which is TIGHT (ET's own VRAM math flags it); a 48 GB card removes any
# OOM risk so this one-shots. ET still reasons at 432 GB/s (the client's SFF).
#
# What this PROVES: the real GGUF MoE read is correct, the roofline/ceiling are
# sane at 432 GB/s, prefix caching really collapses TTFT (the client's lever),
# and the whole pipeline runs clean on real llama-server data — no crashes.
# What it CANNOT prove: the SFF's absolute tok/s (different silicon) or 70 W
# thermals. Those stay bandwidth-scaled estimates.
set -u
export PATH="/usr/local/cuda/bin:$PATH"
HF="${HF:-unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M}"   # client model CLASS (swap to Zane's actual once known)
CTX="${CTX:-32768}"
GPU_BW="${GPU_BW:-432}"                            # client SFF bandwidth (GB/s)
PORT="${PORT:-8080}"
S="${S:-/workspace/llama.cpp/build/bin/llama-server}"
WORK="${WORK:-/workspace}"
URL="http://127.0.0.1:$PORT"

wait_up(){ for _ in $(seq 1 360); do curl -fsS "$URL/props" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

echo "=== [1/5] build llama.cpp if needed ==="
if [ ! -x "$S" ]; then
  apt-get update -y -qq && apt-get install -y -qq build-essential cmake git libcurl4-openssl-dev curl python3 python3-venv
  command -v nvcc >/dev/null || { echo "NO CUDA COMPILER — use the RunPod PyTorch template"; exit 1; }
  mkdir -p "$WORK" && cd "$WORK"
  [ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
  cd llama.cpp
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
  cmake -B build -DGGML_CUDA=ON ${ARCH:+-DCMAKE_CUDA_ARCHITECTURES=$ARCH} >/dev/null
  cmake --build build --config Release -j"$(nproc)" --target llama-server
fi
[ -x "$S" ] || { echo "build failed / set S=/path/to/llama-server"; exit 1; }

echo "=== [2/5] start llama-server (MoE, slots + prefix cache on) ==="
pkill -f "llama-server" 2>/dev/null; sleep 2
"$S" -hf "$HF" --host 127.0.0.1 --port "$PORT" --metrics --slots \
     -ngl 999 --flash-attn on --cache-reuse 256 --ctx-size "$CTX" >/tmp/llama.log 2>&1 &
LPID=$!
trap 'kill $LPID 2>/dev/null' EXIT
echo "   downloading model + loading (first run ~a few min)..."
wait_up || { echo "server didn't start:"; tail -25 /tmp/llama.log; exit 1; }
MODEL=$(find /root/.cache "$WORK" -name '*.gguf' 2>/dev/null | head -1)
echo "   model: $MODEL"

echo "=== [3/5] install ET (monitor + remediation) ==="
cd "$WORK"
[ -d Fixer ] || git clone --depth 1 https://github.com/Ronit-Devan/Fixer >/dev/null 2>&1
python3 -m venv "$WORK/et-venv" 2>/dev/null || true
"$WORK/et-venv/bin/pip" install -q -e "$WORK/Fixer/packages/remediation" -e "$WORK/Fixer/packages/monitor[gpu]" 2>&1 | tail -1
ETPY="$WORK/et-venv/bin/python"

echo "=== [4/5] ET --detect on the REAL MoE, reasoning at the client's 432 GB/s ==="
"$WORK/et-venv/bin/et-monitor" --detect --llama-url "$URL" --model "$MODEL" --gpu-bandwidth "$GPU_BW" 2>&1 | sed 's/^/   /'

echo "=== [5/5] prefix-cache proof: same long prompt COLD vs WARM (the client's lever) ==="
"$ETPY" - "$URL" "$CTX" <<'PYEOF'
import json, sys, urllib.request
url, ctx = sys.argv[1], int(sys.argv[2])
# ~20k-token prompt (well under ctx): repeat a paragraph.
para = ("The memory-bandwidth wall governs single-stream decode while prefill is "
        "dominated by prompt processing. ") * 1200
def gen(prompt, cache):
    body = json.dumps({"prompt": prompt, "n_predict": 32, "cache_prompt": cache, "temperature": 0}).encode()
    req = urllib.request.Request(url + "/completion", body, {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(req, timeout=600)).get("timings", {})
    return t.get("prompt_ms"), t.get("prompt_n"), t.get("cache_n"), t.get("predicted_per_second")
cold_ms, pn, cold_cache, dec = gen(para, True)
warm_ms, _, warm_cache, _ = gen(para, True)   # same prompt again -> prefix cache hit
print(f"   prompt tokens        : {pn}")
print(f"   COLD prefill (TTFT)  : {cold_ms/1000:.2f} s   (cache hit: {cold_cache} tok)")
print(f"   WARM prefill (TTFT)  : {warm_ms/1000:.3f} s  (cache hit: {warm_cache} tok)")
print(f"   decode               : {dec:.1f} tok/s")
if warm_ms > 0 and cold_ms > 0:
    print(f"   >>> prefix cache cut TTFT {cold_ms/max(warm_ms,1):.0f}x  <-- this is the lever ET recommends")
print("   (absolute tok/s is this card, not the 432 GB/s SFF; scale by bandwidth for the SFF.)")
PYEOF

echo "=== DONE — terminate the pod to stop billing ==="
