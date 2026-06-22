# ET: diagnose why your GPU is idle

PyTorch Profiler trace in, root-cause verdict out. Eight verdicts calibrated on real Colab traces, grounded in published systems research.

> **Two products, same idea: attribute GPU idleness to a root cause.**
> - **Training** (`packages/engine`): diagnose a PyTorch Profiler trace. ← this README.
> - **Inference** (`packages/monitor`): a local web app you run on a `llama.cpp`
>   serving box: live GPU + `llama-server` monitoring, idle/decode/KV/throttle
>   verdicts, and a wasted-GPU-$ readout. One command, opens in your browser.
>   See [`packages/monitor/README.md`](packages/monitor/README.md).
>   GTM plan: [`docs/business/GTM.md`](docs/business/GTM.md).
> - **Remediation** (`packages/remediation`): the actuation layer that *applies*
>   the fix and resolves the issue — auto-applying non-disruptive fixes and gating
>   disruptive ones behind human approval, **without ever killing the running
>   workload.** See [`packages/remediation/README.md`](packages/remediation/README.md).

---

## See it work

```
$ gpu-doctor fixtures/dataloader_starved.json

╭────────────────────────────────── Verdict ───────────────────────────────────╮
│ DATALOADER_BOUND  (confidence: 95%)                                          │
│                                                                              │
│ GPU is 16% utilized. The dominant cause is dataloader stalls: 7416ms (99%)   │
│ of GPU idle time overlaps with PyTorch DataLoader activity on the CPU.       │
╰──────────────────────────────────────────────────────────────────────────────╯
                Evidence
┌───────────────────────────────────────┐
│   GPU utilization: 16%                │
│   Total GPU idle: 7481ms              │
│   Dataloader time during idle: 7416ms │
└───────────────────────────────────────┘

Recommended actions:
  1. Increase DataLoader num_workers (try 4 or 8).
  2. Set persistent_workers=True to avoid worker re-spawn.
  3. Set pin_memory=True for faster H2D transfer.
  4. Move expensive preprocessing to a separate process or use an iterable dataset.
  5. Profile your __getitem__ and look for slow image decode or disk reads.
```

---

## Install

```bash
git clone https://github.com/devan-p/ET
cd ET/packages/engine
uv sync
uv run gpu-doctor ../../fixtures/dataloader_starved.json
```

For pip and wheel installation see `packages/engine/INSTALL.md`.

---

## What it detects

| Verdict | Status | When it fires | Research foundation |
|---|---|---|---|
| HEALTHY | Active | GPU util ≥ 70%, no dominant suspect |  |
| DATALOADER_BOUND | Active | DataLoader patterns ≥ 20% of GPU idle time | MinatoLoader (arXiv 2509.10712) |
| PCIE_BOUND | Active | Memcpy ≥ 50% of GPU-active time (or ≥ 30% of idle) |  |
| KERNEL_LAUNCH_BOUND | Active | >50% of kernels < 50µs, low util |  |
| NCCL_BOUND | Active | NCCL collectives (AllReduce, AllGather, …) ≥ 30% of GPU idle time |  |
| CHECKPOINT_BOUND | Active | torch.save dominates idle time | DataStates-LLM (HPDC '24, arXiv 2406.10707) |
| SYNC_BOUND | Active | Host sync calls ≥ 25% of GPU idle time |  |
| UNKNOWN | Active | Low util, no clear pattern |  |

For `DATALOADER_BOUND`, the engine also detects head-of-line blocking; one slow sample holding up the worker pool; using the p99/median duration ratio from MinatoLoader.

---

## How it works

- **Merged-interval math.** Overlapping kernel and `gpu_memcpy` events are merged before any computation. Naive duration sums double-count concurrent ops across streams.
- **Idle-window attribution by overlap.** Each gap in GPU activity is overlapped with CPU-side event patterns (DataLoader names, NCCL ops, Memcpy calls) to attribute stall time.
- **Specific causes beat generic causes.** Memcpy, NCCL, and checkpoint each have dedicated thresholds; when they fire, they win over DataLoader, which is a wrapper that can contain all of them.
- **Per-rule decision log.** Every threshold check is evaluated and logged in order. Visible in `--explain`.

---

## Remediation: apply the fix, not just advise it

`packages/remediation` turns a diagnosed root cause into an **action**, behind a
hard safety model. It plugs into the live monitor exactly parallel to the alert
hook (`Monitor.tick()` → diagnosis → `RemediationManager.observe()`), so
detection/diagnosis are untouched.

- **Two classes.** Every fix is **NON-DISRUPTIVE** (power/clock limits, MPS/MIG,
  freeing stale cache, killing *only* orphaned PIDs, re-nicing loader workers) or
  **DISRUPTIVE** (needs a job / `llama-server` restart).
- **Non-disruptive auto-applies unattended** — but never blind: through
  `classify → guard → apply → verify recovery in a bounded window → confirm or
  AUTO-ROLLBACK`.
- **Disruptive never auto-fires** — it only ever opens a human-gated approval
  request, after checkpoint + request-drain.
- **The running workload is never killed by a non-disruptive path** — enforced
  structurally (a non-disruptive action can't carry a restart kind; every
  process-touching call refuses a protected PID).
- **Circuit breaker** trips auto-apply to advise-only on repeated failure, flap,
  or rate-cap breach. A **global kill-switch** (`off`/`advise`/`dry_run`/`auto`)
  forces advise-only for ops, set via a first-run setup wizard, the CLI
  (`et-remediation mode advise`), or the dashboard. Every action is **audited**.
- **Two backends, one interface:** a DataCenter actuator (nvidia-smi/NVML/DCGM,
  K8s/Slurm) and a llama.cpp actuator (drain + restart with tuned flags).

Full details and the safety model: [`packages/remediation/README.md`](packages/remediation/README.md).

## See the engine's reasoning

```
$ gpu-doctor fixtures/dataloader_starved.json --explain

Detector decisions:
  ✗ healthy_85                value=0.16  threshold=0.85   skipped
  ✗ healthy_70_no_dominant    value=0.16  threshold=0.70   skipped
  ✗ pcie_ratio_50             value=0.01  threshold=0.50   skipped
  ✗ checkpoint_25             value=0.01  threshold=0.25   skipped
  ✗ kernel_launch_tiny        value=0.35  threshold=0.50   skipped
  ✗ nccl_bound_30             value=0.00  threshold=0.30   skipped
  ✓ dataloader_fallback       value=0.99  threshold=0.20   FIRED
```

Every rule is evaluated regardless of verdict. You can see exactly why a rule did or did not fire.

---

## CLI

Bare invocation; no subcommand needed:

```bash
gpu-doctor trace.json
```

JSON output for scripting:

```bash
gpu-doctor trace.json --json | jq '.verdict'
```

Top events by duration:

```bash
gpu-doctor trace.json --top-events 10
```

Full reasoning including per-rule decision log:

```bash
gpu-doctor trace.json --explain
```

Portable markdown report:

```bash
gpu-doctor report trace.json --output diagnosis.md
```

---

## Research foundation

We read systems papers and ship the insights as detectors.

**Active in v0.3:**
- **MinatoLoader** (arXiv 2509.10712): dataloader head-of-line blocking; one slow sample stalls the entire worker pool. Informs `DATALOADER_BOUND` HoL detection and recommended actions.
- **DataStates-LLM** (HPDC '24, arXiv 2406.10707): synchronous checkpoint overhead as a first-class GPU stall source. Informs `CHECKPOINT_BOUND` threshold and confidence model.
- **NCCL idle attribution**: collective communication (AllReduce, AllGather, ReduceScatter, Broadcast) measured as overlap with GPU-idle windows at ≥30%; dedicated `nccl_bound_30` detector in `packages/engine/src/gpu_doctor_engine/detectors/nccl.py`.

**Planned:**
- **eGPU** (HCDS '25, DOI 10.1145/3723851.3726984): eBPF/PTX agent layer for continuous, kernel-level GPU observability.
- **Marconi** (MLSys '25, arXiv 2411.19379): prefix cache detector for LLM inference workloads.
- **SpotServe** (ASPLOS '24, arXiv 2311.15566): spot-GPU preemption and resilience detector.

---

## Status

v0.3 engine. Eight verdicts calibrated on real Colab traces. Kubernetes agent and web UI in progress. Guarded auto-remediation layer (`packages/remediation`).

Hardened for **accuracy, efficiency, and scale**: predictive early-warning (throttle / KV-saturation / OOM caught *before* they land), adaptive sampling (~78% fewer GPU reads on a stable box — minimal overhead next to your workload), and true multi-GPU / fleet support (per-GPU diagnosis + remediation, fleet-aggregate idle cost, per-(node,gpu) blast-radius). 432 tests across engine (140), agent (111), monitor (77), remediation (95), and web/api (9); CI green.

---

## Development

See `packages/engine/INSTALL.md` for environment setup.

Run tests:

```bash
cd packages/engine
uv run pytest
```

Lint:

```bash
uv run ruff check src/ tests/
```

Add a new fixture trace: generate one in Colab, drop it in `fixtures/`, add an entry to the parametrize list in `tests/test_real_traces.py`.

---

## License

MIT.
