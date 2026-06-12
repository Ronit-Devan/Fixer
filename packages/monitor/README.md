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
| **Healthy** | Well used, nothing to do |  |

It also shows live charts (utilization, VRAM, tokens/sec, KV cache) and a
session **idle-fraction + projected monthly idle cost**.

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
