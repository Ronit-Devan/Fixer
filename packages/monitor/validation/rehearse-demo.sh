#!/usr/bin/env bash
# Dress rehearsal for the client demo: run the FULL ET product live against the
# client's four benchmark scenarios (short chat / 8K cold / 30K cold / 30K warm)
# on a real MoE, and print ET's live verdict after each — plus the client-style
# e2e tok/s table. Leaves the ET dashboard up for screenshots.
#
#   curl -fsSL https://raw.githubusercontent.com/Ronit-Devan/Fixer/main/packages/monitor/validation/rehearse-demo.sh | bash
#
# Rent a 48 GB GPU (L40S / A6000). ET reasons at the client's 432 GB/s SFF
# bandwidth. NOTE: this card prefills ~2-3x faster than the SFF, so the 8K-cold
# scenario may come back "healthy" here while being prefill-bound on the client
# box — the output annotates that honestly.
set -u
export PATH="/usr/local/cuda/bin:$PATH"
HF="${HF:-unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M}"
CTX="${CTX:-32768}"
GPU_BW="${GPU_BW:-432}"
PORT="${PORT:-8080}"
MON_PORT="${MON_PORT:-7070}"
S="${S:-/workspace/llama.cpp/build/bin/llama-server}"
WORK="${WORK:-/workspace}"
URL="http://127.0.0.1:$PORT"
MON="http://127.0.0.1:$MON_PORT"

wait_url(){ for _ in $(seq 1 360); do curl -fsS "$1" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

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
apt-get install -y -qq python3-venv >/dev/null 2>&1 || true

echo "=== [2/5] start llama-server (slots + metrics + prefix reuse on) ==="
pkill -f "llama-server" 2>/dev/null; sleep 2
"$S" -hf "$HF" --host 127.0.0.1 --port "$PORT" --metrics --slots \
     -ngl 999 --flash-attn on --cache-reuse 256 --ctx-size "$CTX" >/tmp/llama.log 2>&1 &
LPID=$!
trap 'kill $LPID 2>/dev/null' EXIT
echo "   model download + load (first run takes a few min)..."
wait_url "$URL/props" || { echo "server didn't start:"; tail -25 /tmp/llama.log; exit 1; }
MODEL=$(find /root/.cache "$WORK" -name '*.gguf' 2>/dev/null | head -1)

echo "=== [3/5] install ET + detect (client bandwidth: $GPU_BW GB/s) ==="
cd "$WORK"
[ -d Fixer ] || git clone --depth 1 https://github.com/Ronit-Devan/Fixer >/dev/null 2>&1
(cd Fixer && git pull -q 2>/dev/null || true)
python3 -m venv "$WORK/et-venv" 2>/dev/null || true
"$WORK/et-venv/bin/pip" install -q -e "$WORK/Fixer/packages/remediation" -e "$WORK/Fixer/packages/monitor[gpu]" 2>&1 | tail -1
"$WORK/et-venv/bin/et-monitor" --detect --llama-url "$URL" --model "$MODEL" --gpu-bandwidth "$GPU_BW" 2>&1 | sed 's/^/   /'

echo "=== [4/5] start the live ET monitor (dashboard on :$MON_PORT) ==="
pkill -f "et-monitor" 2>/dev/null; sleep 1
"$WORK/et-venv/bin/et-monitor" --llama-url "$URL" --host 0.0.0.0 --port "$MON_PORT" \
    --no-browser >/tmp/et-monitor.log 2>&1 &
MPID=$!
trap 'kill $LPID $MPID 2>/dev/null' EXIT
wait_url "$MON/healthz" || { echo "monitor didn't start:"; tail -20 /tmp/et-monitor.log; exit 1; }
sleep 8  # let it collect min samples

echo "=== [5/5] drive the four client scenarios, polling ET's live verdict ==="
"$WORK/et-venv/bin/python" - "$URL" "$MON" <<'PYEOF'
import json, sys, time, urllib.request

url, mon = sys.argv[1], sys.argv[2]
PARA = ("The memory-bandwidth wall governs single-stream decode while prefill "
        "is dominated by prompt-processing compute. ")

def gen(prompt, n=420):
    body = json.dumps({"prompt": prompt, "n_predict": n, "cache_prompt": True,
                       "temperature": 0}).encode()
    req = urllib.request.Request(url + "/completion", body,
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    t = r.get("timings", {})
    ttft = (t.get("prompt_ms") or 0) / 1000
    wall = ttft + (t.get("predicted_ms") or 0) / 1000
    return {
        "ttft": ttft, "wall": wall,
        "e2e": (t.get("predicted_n") or 0) / wall if wall > 0 else 0,
        "decode": t.get("predicted_per_second") or 0,
        "cache": t.get("cache_n") or 0,
        "prompt_n": t.get("prompt_n") or 0,
    }

def verdict():
    time.sleep(6)  # let the monitor's ticks land
    d = json.load(urllib.request.urlopen(mon + "/api/diagnosis", timeout=10))
    m = d.get("metrics", {})
    return {"verdict": d.get("verdict", "?"), "title": d.get("title", ""),
            "hit": m.get("prefix_cache_hit_rate"), "ttft": m.get("ttft_s")}

nonce = str(time.time_ns())  # make cold prompts truly cold across reruns
scenarios = [
    ("Short chat",       "Explain GPUs briefly. " + nonce,   "not prefill-bound"),
    ("8K ctx, cold",     nonce + " " + PARA * 420,           "prefill-bound on the 432 GB/s SFF (this card may prefill it in <2s and call it healthy — annotate, don't oversell)"),
    ("30K ctx, cold",    nonce + "! " + PARA * 1500,         "PREFILL_BOUND"),
    ("30K ctx, warm",    None,                               "not prefill-bound (prefix cache hit)"),
]
rows, prev_prompt = [], None
for i, (name, prompt, expect) in enumerate(scenarios):
    if prompt is None:
        prompt = prev_prompt  # warm = replay the 30K prompt
    else:
        prev_prompt = prompt
    if i:
        print(f"   ... 40s pause (flush the analyzer window) ...")
        time.sleep(40)
    r = gen(prompt)
    v = verdict()
    rows.append((name, r, v, expect))
    print(f"   {name:16s} TTFT {r['ttft']:6.2f}s  e2e {r['e2e']:6.1f} tok/s  "
          f"decode {r['decode']:6.1f}  cache_hit {r['cache']:>6}  "
          f"ET verdict: {v['verdict']}")

print()
print("=" * 76)
print(f"{'Scenario':<16} {'TTFT':>7} {'e2e tok/s':>10} {'decode':>8} {'ET verdict':>18}   expected")
print("-" * 76)
for name, r, v, expect in rows:
    print(f"{name:<16} {r['ttft']:>6.2f}s {r['e2e']:>10.1f} {r['decode']:>8.1f} "
          f"{v['verdict']:>18}   {expect}")
print("=" * 76)
cold, warm = rows[2][1], rows[3][1]
if warm["ttft"] > 0:
    print(f">>> 30K prefix cache: TTFT {cold['ttft']:.2f}s -> {warm['ttft']:.3f}s "
          f"({cold['ttft'] / max(warm['ttft'], 1e-3):.0f}x), "
          f"e2e {cold['e2e']:.1f} -> {warm['e2e']:.1f} tok/s")
print(">>> Reminder: absolute tok/s is THIS card; the client's SFF has ~half the")
print(">>> bandwidth (decode ~= scale by 0.5) and slower prefill (TTFTs larger).")
PYEOF

POD="${RUNPOD_POD_ID:-<POD_ID>}"
echo ""
echo "=== dashboard is LIVE for screenshots: https://${POD}-${MON_PORT}.proxy.runpod.net ==="
echo "=== (report page: same URL + /report). Terminate the pod when done. ==="
echo "Press Ctrl-C to stop servers (or just terminate the pod)."
wait $MPID 2>/dev/null || true
