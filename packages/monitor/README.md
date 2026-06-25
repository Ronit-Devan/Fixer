# ET monitor: is your GPU actually being used?

A **local web app** you run on the machine serving a model with `llama.cpp`. It
watches the GPU and `llama-server` live, tells you *why* the card is idle or
under-used, and estimates the dollars you're spending on idle time.

No cloud, no account, no build step. One command, opens in your browser.

---

## Requirements

A single-GPU `llama.cpp` box needs exactly four things; check each before you start:

| Need | Why | Verify |
|---|---|---|
| **NVIDIA GPU + driver** | live utilization / VRAM / power / clock readings (via NVML) | `nvidia-smi` prints your card |
| **Python 3.10+** | runs the monitor | `python --version` (Windows) / `python3 --version` |
| **`llama.cpp` `llama-server`** | the inference server being watched | `llama-server --version` |
| **Git** | to fetch the repo (or download the ZIP) | `git --version` |

No CUDA toolkit, compiler, or build step is needed for the monitor itself — it
*reads* the driver, it doesn't compile kernels. A single GPU is the common case
here, and it's exactly where the **decode roofline** earns its keep: one stream
can't push utilization to 90%, so "40% utilized" is meaningless without it.

## Install & run (single GPU + llama.cpp)

The `run.sh` / `run.bat` script *is* the installer: it creates a private
virtualenv, installs the app with NVIDIA telemetry, and opens the dashboard.
No `uv`, no Docker, no global Python pollution. Four steps, ~30 seconds.

**1. Get the code**

```bash
git clone https://github.com/Ronit-Devan/Fixer
cd Fixer/packages/monitor
```

**2. Start `llama-server` with metrics** — in its own terminal, left running

The monitor reads llama.cpp's Prometheus metrics, which the server only exposes
when you pass `--metrics`:

```bash
llama-server -m model.gguf -ngl 999 --port 8080 --metrics
```

`-ngl 999` offloads **all** layers onto your single GPU; the monitor flags it if
any are stuck on the CPU (the #1 single-GPU throughput bug). Default port `8080`
is what the monitor looks for.

**3. Launch the monitor**

macOS / Linux:

```bash
./run.sh --gpu-price 0.50
```

Windows:

```bat
run.bat --gpu-price 0.50
```

First run creates `.venv/`, installs `et-gpu-monitor` with the `[gpu]` extra
(`nvidia-ml-py`; it falls back to `nvidia-smi`, then mock data, if that wheel
won't build), and opens `http://localhost:7070` in your browser. `--gpu-price`
is your GPU cost in $/hour — it turns on the "wasted on idle" dollar readout;
omit it if you don't care. If your `llama-server` is elsewhere, add
`--llama-url http://localhost:8081`.

**4. Turn on the decode roofline** (recommended, one-time)

```bash
./run.sh --detect          # Windows: run.bat --detect
```

This probes `llama-server` `/props` + the model GGUF + the GPU, prints your
single-stream tok/s ceiling, and saves `~/.et/workload.json` so MBU / "% of
ceiling" / partial-offload diagnosis runs every tick. It also **auto-runs on
first launch** whenever `llama-server` is reachable, so on a fresh box the
roofline is usually already on — run it explicitly to see the ceiling number, or
when you need the flags below:

- model path not exposed by `/props`? add `--model /path/to/model.gguf`
- card not in the bandwidth table? add `--gpu-bandwidth <GB/s>`

That's the whole install. The dashboard updates live; **Ctrl-C** to stop.

**No GPU handy? See it work anyway:**

```bash
./run.sh --demo --gpu-price 0.50
```

The demo plays a scripted inference timeline so the dashboard cycles through
every verdict (idle → decode-bound → memory headroom → KV pressure →
throttling → healthy) — no GPU or model required.

---

## Point it at your llama-server

llama.cpp's server only exposes metrics when you start it with `--metrics`:

```bash
llama-server -m model.gguf --port 8080 --metrics
```

The monitor looks for `llama-server` at `http://localhost:8080` by default. If
yours is elsewhere:

```bash
./run.sh --llama-url http://localhost:8081 --gpu-price 0.50
```

If `llama-server` isn't reachable, the monitor still runs in **GPU-only mode**
(utilization, memory, power, idle detection from NVML); you just don't get the
request/KV-cache attribution.

---

## The decode roofline: "40% of *what*?"

`nvidia-smi` utilization is **not** throughput. On a single GPU serving one
stream, decode is *memory-bandwidth bound* — utilization plateaus well below
100% even when tokens/sec is already optimal. So "my GPU is at 40%" is ambiguous.

Run detection once and the monitor answers it:

```bash
et-monitor --detect      # probes llama-server /props + the model GGUF + the GPU
```

That writes `~/.et/workload.json` (model size, layer count, GPU memory
bandwidth); the monitor then computes, every tick:

- **MBU** (memory-bandwidth utilization) and the **single-stream tok/s ceiling**,
- **% of ceiling** you're actually hitting — the honest answer to "40% of what?",
- **partial offload** — if `-ngl` left layers on the CPU (the #1 single-GPU
  throughput bug), with a fix that's VRAM-fit-checked.

It splits the old "decode-bound" verdict three ways instead of one:

- **At the single-stream wall** (MBU near the limit): physics, not a bug — you
  can't push util to 90% at concurrency 1. The lever is batching / speculative
  decoding / a faster quant; chasing util% here is a vanity metric.
- **Under-batched**: real concurrent demand is queueing — `--parallel N
  --cont-batching`.
- **Host-bound**: well below the bandwidth wall on a fully-offloaded model —
  something host-side (threads, batch size, flash-attention off) is the limit.

If your card isn't in the bandwidth table, pass `--gpu-bandwidth <GB/s>`. The
monitor also **auto-detects on first run** when it can reach llama-server, so on
a fresh box the roofline is usually on without any extra step. The dashboard
shows a *throughput-vs-ceiling* panel (tok/s, MBU bar with the single-stream wall
marked, % of ceiling, and the plain-language reason).

## Slack / webhook alerts

Get pinged when the GPU needs attention; no need to watch the dashboard.

```bash
./run.sh --gpu-price 0.50 \
  --slack-webhook https://hooks.slack.com/services/XXX/YYY/ZZZ \
  --host-label sf-blackwell-01
```

- Create a Slack **Incoming Webhook** (Slack → Apps → Incoming Webhooks → pick a
  channel) and paste the URL into `--slack-webhook`.
- `--webhook <url>` posts the full diagnosis JSON to any endpoint (PagerDuty,
  n8n, your own bot).
- It alerts when the box enters a **warn/crit** state (KV pressure, throttling)
  and when the GPU is **idle past `--alert-idle-min`** (default 10 min). It sends
  **once per episode** + a "recovered" message; never a stream of repeats.
- Tune with `--alert-idle-min`, `--alert-cooldown-min`, `--no-alert-recovery`.

## What it tells you

| Verdict | Meaning | What to do |
|---|---|---|
| **Idle (no requests)** | GPU sitting idle, no inference traffic | Quantify the cost; co-locate batch work; unload-on-idle |
| **Model partly on CPU** | `-ngl` left layers on the CPU; throughput capped far below the card | Restart with `-ngl 999` (the monitor checks it fits in VRAM first) |
| **Decode bandwidth-bound** | Serving below saturation; with a roofline, split into single-stream wall (physics) vs under-batched vs host-bound | Batch (`--parallel N --cont-batching`), speculative decoding, or fix the host-side limit — per the sub-reason |
| **Memory under-used** | Lots of VRAM free while in use | Bigger/higher-precision model, larger context, more parallel slots |
| **KV cache pressure** | Cache near full / requests deferred | Tune `--ctx-size`/`--parallel`; prefix caching; this is your scale-out signal |
| **Throttling** | SM clock dragged down under load | Cooling/airflow; check power limit |
| **VRAM pressure** | VRAM climbing toward OOM (predicted) | Free/again-cache before it crashes; lower ctx/parallel |
| **Healthy** | Well used, nothing to do |  |

It also shows live charts (utilization, VRAM, tokens/sec, KV cache) and a
session **idle-fraction + projected monthly idle cost**.

### Catches problems *before* they land (predictive)

Beyond the reactive verdicts above, the monitor fits a trend to the recent window
and projects when a metric will cross a danger line — so it warns with lead time:

- **Throttle imminent** — temperature climbing toward the throttle point (or SM
  clock sliding) *before* tokens/sec are lost.
- **KV saturation imminent** — cache fill-rate projects it will hit the ceiling
  and start deferring requests soon.
- **OOM imminent** — VRAM growth trend projects an out-of-memory crash, the one
  failure that kills the workload.

Predicted verdicts carry `predicted: true` and a `horizon_s` (estimated seconds
until the event), which lets the remediation layer act pre-emptively.

### Multi-GPU & fleets

The monitor tracks **every GPU on the box** (a DGX with 8 cards, not just card 0):
per-GPU history, diagnosis, and remediation, with a **fleet-aggregate** idle-cost
readout (so an 8-GPU box reports 8 cards' idle dollars, not one). `GET /api/state`
carries a `gpus[]` array; the report breaks down per card. Each GPU remediates
independently (its own circuit breaker + approvals), and a shared blast-radius cap
limits how many cards auto-remediate at once. Pass `--host-label` (or it uses the
hostname) so a fleet's events are distinguishable.

### Very low overhead (non-negotiable)

This runs next to your production workload, so it must not perturb it. The loop
**adaptively backs off** when the box is provably quiescent (healthy or plainly
idle) — up to ~78% fewer GPU reads over a stable window — and snaps back to the
fast rate the instant anything changes or a trend looks risky. It prefers NVML
(no subprocess) over `nvidia-smi`, caches the GPU count, walks only the trailing
window per tick, and self-reports its own per-tick cost at `snapshot().perf`.

### Shareable report

Open `/report` (or the "View / print report" link in the dashboard footer) for a
clean one-page summary; idle %, average utilization, estimated idle cost per
month/year, and a breakdown of where the GPU's time went. Hit **Print / Save as
PDF** to send it to a stakeholder. Raw numbers are at `GET /api/report`.

---

## Install as a package (optional)

```bash
pip install -e ".[gpu]"   # [gpu] adds nvidia-ml-py; omit to use nvidia-smi
et-monitor --gpu-price 0.50
```

GPU backend order: `pynvml` → `nvidia-smi` → mock. Whichever works first wins;
the chosen backend is logged at startup and shown in the dashboard footer.

---

## Auto-remediation (optional)

With `packages/remediation` installed, the monitor can *apply* the fix, not just
advise it. It's **off by default**; enable it per the operating mode:

```bash
et-monitor --remediation-setup                 # first-run wizard: pick the mode
et-monitor --remediation-mode advise           # recommend only (never touches the box)
et-monitor --remediation-mode auto             # auto-apply NON-DISRUPTIVE fixes
et-monitor --remediation-mode auto --remediation-audit-log ~/.et/remediation.jsonl
```

- **NON-DISRUPTIVE fixes** (power/clock limits, MPS/MIG, freeing stale cache,
  killing only orphaned PIDs) auto-apply through a guarded path: apply → watch a
  bounded window for recovery → confirm or **auto-rollback**. The running
  `llama-server` / training job is **never** killed by this path.
- **DISRUPTIVE fixes** (restart `llama-server` with tuned flags) **never**
  auto-fire — they appear as an approval card in the dashboard, applied only
  after you approve (and a request-drain). The restart is **derived from the
  roofline**: raise `-ngl` for partial offload (VRAM-fit-checked), raise
  `--parallel`/`--cont-batching` only when there's real concurrent demand, and
  recovery is judged on **tokens/sec** (not utilization, which single-stream
  can't move). A box at the physical single-stream wall is left alone (no
  pointless restart) and just explained.
- A **circuit breaker** trips auto-apply to advise-only on repeated failure /
  flap / rate-cap breach.

**Disable auto-apply for ops (the kill-switch):**

```bash
et-remediation mode advise        # or: off
curl -XPOST localhost:7070/api/remediation/mode -d '{"mode":"advise"}'
```

The dashboard shows a remediation panel (mode selector, breaker state, pending
approvals, recent actions). Full safety model:
[`../remediation/README.md`](../remediation/README.md).

## How it fits ET

This is ET's **inference** product. The training engine (`packages/engine`)
diagnoses PyTorch Profiler traces; this monitor diagnoses a live `llama.cpp`
serving box from NVML + the server's Prometheus metrics; no PyTorch, no trace
file. Same idea (attribute GPU idleness to a root cause), different workload.

## Dev

```bash
cd packages/monitor
pip install -e ".[dev]"
pytest -q
```
