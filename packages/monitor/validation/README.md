# ET validation harness — prove it works on real hardware

Four scripts that turn "should raise TPS" into "measured to raise TPS." Built to
run on a cheap rented GPU (Runpod / Vast / Lambda, ~$0.40–$2/hr) before you ever
touch a production box.

| Script | What it proves | Needs a GPU? | Risk |
|---|---|---|---|
| `check-flags.sh` | The tuned flags exist in *this* llama.cpp build (kills version skew) | no (just the binary) | none |
| `bench-tps.sh` | Real decode tok/s from a running server | a running llama-server | none (read) |
| `dryrun.sh` | Diagnoses a live server + prints the exact fix — **touches nothing** | a running llama-server | **none** |
| `validate-e2e.sh` | Plants a misconfig, confirms ET catches it, applies the fix, benchmarks before/after | yes | local box only |

## The 10-minute path on a rented GPU

1. **Rent a single GPU** (Runpod "RTX 4090" or "A100" pod is fine). In its terminal:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/Ronit-Devan/Fixer/main/packages/monitor/setup-on-gpu.sh | bash
   ```
   This builds llama.cpp (CUDA), pulls a small model, and starts llama-server.

2. **Run the proof** (from the cloned repo on the pod):
   ```bash
   cd /workspace/Fixer/packages/monitor/validation
   SERVER=/workspace/llama.cpp/build/bin/llama-server \
   MODEL=$(ls /root/.cache/llama.cpp/*.gguf | head -1) \
   ./validate-e2e.sh
   ```
   You get a table:
   ```
   baseline (partial offload) : 14.2 tok/s
   fixed (full offload + FA)  : 58.9 tok/s
   speedup                    : 4.15x
   ET diagnosed the problem   : yes
   PASS — ET correctly diagnosed and the fix measurably raised tok/s.
   ```

3. **(Optional) test speculative decoding** — add a small same-family draft model:
   ```bash
   DRAFT_MODEL=/workspace/models/qwen2.5-0.5b-q4_k_m.gguf  ... ./validate-e2e.sh
   ```

## Before touching Zane's box (zero risk, costs nothing)

Point the read-only dry-run at his server. It only does HTTP GET on `/props` and
`/metrics` — no restart, no writes:
```bash
./check-flags.sh                 # on a box with his llama-server binary
./dryrun.sh --url http://ZANE_HOST:8080
```
The output tells you the one thing that decides the $10k:
- **"WIN AVAILABLE: model is partially on CPU"** → big lever, `-ngl 999` likely multiplies his TPS.
- **"AT THE WALL"** → he's already maxed single-stream; only spec-decode / smaller quant / batching can move it (and not to any guaranteed number).
- **"well-configured"** → there may be no easy win; read MBU/ceiling for remaining headroom.

## Honest scope

These prove **the tool reliably diagnoses and that its fixes raise tok/s on real
hardware, with correct flags for the build.** They do **not** guarantee Zane's
specific box has a fixable problem — `dryrun.sh` against *his* server is what
answers that. A "no win" result is a real finding, not a failure.
