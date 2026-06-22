# ET monitor: is your GPU actually being used?

A **local web app** you run on the machine serving a model with `llama.cpp`. It
watches the GPU and `llama-server` live, tells you *why* the card is idle or
under-used, and estimates the dollars you're spending on idle time.

No cloud, no account, no build step. One command, opens in your browser.

---

## Run it (30 seconds)

**macOS / Linux**

```bash
cd packages/monitor
./run.sh --gpu-price 0.50
```

**Windows**

```bat
cd packages\monitor
run.bat --gpu-price 0.50
```

That creates a virtual environment, installs the app, and opens
`http://localhost:7070` in your browser. `--gpu-price` is your GPU cost in
$/hour. It turns on the "wasted on idle" readout (leave it off if you don't
care about the dollar figure).

**No GPU handy? See it work anyway:**

```bash
./run.sh --demo --gpu-price 0.50
```

The demo plays a scripted inference timeline so the dashboard cycles through
every verdict (idle → decode-bound → memory headroom → KV pressure →
throttling → healthy).

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
| **Decode bandwidth-bound** | Serving at low concurrency; util plateaus below 100% | Batch concurrent requests (`--parallel N`); speculative decoding |
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
- **DISRUPTIVE fixes** (restart `llama-server` with tuned `-ngl`/`-t`/`-b`/cache
  flags) **never** auto-fire — they appear as an approval card in the dashboard,
  applied only after you approve (and a request-drain).
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
