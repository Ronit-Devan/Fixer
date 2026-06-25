# ET Inference Monitor: setup

Runs on the machine serving the model. Watches the GPU + `llama-server` live and
shows it in a browser. Runs entirely locally; nothing leaves the machine.

---

## 1. Start llama-server with metrics on

The monitor reads `llama-server`'s metrics endpoint, which is **off by default**.
Add `--metrics` when you launch it:

```bash
llama-server -m your-model.gguf --host 0.0.0.0 --port 8080 --metrics
```

(If llama-server is already running without `--metrics`, the monitor still works
in GPU-only mode; you just won't get request/KV-cache detail.)

## 2. Start the monitor

**Linux / macOS**

```bash
cd packages/monitor
./run.sh --gpu-price 0.50 --llama-url http://localhost:8080
```

**Windows**

```bat
cd packages\monitor
run.bat --gpu-price 0.50 --llama-url http://localhost:8080
```

- Set `--gpu-price` to the GPU's real cost in **$/hour** (rental rate, or
  amortized purchase). It powers the "estimated cost of idle" readout. Omit it
  if you don't want a dollar figure.
- It opens `http://localhost:7070` in the browser automatically.

That's it. Leave it running.

> **Tip — turn on the decode roofline.** Run `./run.sh --detect` once (it probes
> llama-server, the model GGUF, and the GPU) to enable MBU / single-stream tok/s
> ceiling / partial-offload diagnosis — so "40% utilized" becomes "40% of the
> card's ceiling (fixable)" vs "at the single-stream wall (physics)". The monitor
> also auto-detects this on first run when it can reach llama-server.

## 3. View from another computer (optional)

To open the dashboard from your laptop instead of the GPU box, bind to the
network and use the box's LAN IP:

```bash
./run.sh --host 0.0.0.0 --port 7070 --gpu-price 0.50
```

Then browse to `http://<gpu-box-ip>:7070` from any machine on the same network.
(Only do this on a trusted network; there's no auth on the dashboard.)

---

## Get Slack alerts (optional, recommended)

So no one has to watch the dashboard:

1. In Slack: **Apps → Incoming Webhooks → Add to Slack**, pick a channel, copy
   the webhook URL.
2. Add it to the run command:

```bash
./run.sh --gpu-price 0.50 \
  --slack-webhook https://hooks.slack.com/services/XXX/YYY/ZZZ \
  --host-label sf-blackwell-01
```

You'll get a message when the GPU sits idle too long, hits KV-cache pressure, or
throttles; once per episode, plus a "recovered" note. No spam. Tune the idle
threshold with `--alert-idle-min N` (minutes).

## Keep it running after logout / reboot

### Linux (systemd); recommended

A ready-made unit is in `deploy/et-monitor.service`. Edit the paths/price inside
it, then:

```bash
sudo cp deploy/et-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now et-monitor
systemctl status et-monitor      # check it's running
journalctl -u et-monitor -f      # follow logs
```

### Windows

Simplest: leave the `run.bat` window open. To auto-start on login, put a
shortcut to `run.bat` (with your args) in the Startup folder
(`Win+R` → `shell:startup`).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Dashboard says "llama-server: not found" | Start it with `--metrics`; check `--llama-url` matches its port |
| No MBU / throughput-vs-ceiling panel | No workload spec yet — run `./run.sh --detect`; if your card's bandwidth is unknown, add `--gpu-bandwidth <GB/s>` |
| "backend: mock" in the header | No NVIDIA GPU detected; install the driver, or `pip install nvidia-ml-py`; `nvidia-smi` must work |
| Page won't load | Check the port isn't taken; try `--port 7071` |
| Want a quick look without a GPU | `./run.sh --demo --gpu-price 0.50` plays a scripted timeline |

Backend selection order is `pynvml` → `nvidia-smi` → mock; the chosen one shows
in the dashboard header.
