# ET remediation: from *advising* a fix to *applying* it — safely

ET's engine and monitor detect a GPU utilization drop and diagnose the root
cause. This package is the **actuation layer**: it consumes a diagnosed root
cause and *resolves* it — applying non-disruptive fixes unattended, and gating
disruptive ones behind human approval — **without ever terminating the running
compute task.**

```
detect → diagnose → [ remediation layer ] → verify → confirm / auto-rollback
                          │
            classify → guard → apply → watch recovery
```

## The safety model (read this first)

Every remediation is classified into one of two classes, and the class decides
everything:

| | **NON-DISRUPTIVE** | **DISRUPTIVE** |
|---|---|---|
| examples | power/clock limits, MPS/MIG tuning, freeing stale cache, killing **only** orphaned PIDs, re-nicing loader workers | restart a training job, restart `llama-server` with new flags |
| touches the live workload? | **never** | yes (so it is fenced) |
| how it runs | **auto-applies unattended** through the guarded path | **only** as a human-gated approval request, after checkpoint + drain |
| fired automatically? | yes — but never blind | **never** |

**The #1 invariant — the running compute task is never killed by a
non-disruptive path — is structural, not a convention:**
- a `NON_DISRUPTIVE` `ActionSpec` *cannot* carry a workload-lifecycle kind
  (restart/drain) — rejected at construction and by the classifier;
- every process-touching actuator call (`renice`, orphan-kill, free-stale)
  funnels through `Actuator.guard_protected()`, which refuses any PID in the
  `ProtectedWorkload` set;
- disruptive restarts drain in-flight requests (waiting, never cancelling) and
  only run after explicit approval.

## The guarded auto-apply path

For a non-disruptive fix in `auto` mode, `RemediationManager.observe()` runs:

1. **classify** — confirm the action is non-disruptive;
2. **policy gate** — kill-switch/mode, per-class enable, protected-workload
   presence, blast-radius (one in-flight action per node);
3. **circuit-breaker gate** — see below;
4. **apply** — idempotent; snapshots prior state for rollback;
5. **verify** — watch a bounded telemetry window (`verify_window_s`, default 30s)
   for recovery (e.g. SM clock climbs back after a throttle fix);
6. **confirm or AUTO-ROLLBACK** — recovered → confirm; window elapsed without
   recovery → revert to the snapshot automatically.

Every step is written to the **audit log** (in-memory ring + optional JSONL):
trigger, decision, apply, verify result, rollback.

**Verify only on post-fix telemetry.** Recovery is judged solely on samples
taken *after* the fix was applied (filtered by timestamp), and only once at least
`min_verify_samples` of them exist — so a rolling window that still contains
pre-fix readings can never cause a false confirm or false rollback, and a single
noisy reading can't drive the decision. Recovery predicates require a real,
attributable improvement (e.g. the clock both cleared a floor *and* gained), not
just a window that happens to look healthy.

**Debounced triggering.** An actionable cause must persist for
`trigger_debounce` consecutive observations before anything fires, so a one-tick
blip never actuates. Paired with the monitor's *predictive* verdicts (which warn
with lead time), the debounce costs nothing in responsiveness.

## The circuit breaker

Auto-apply trips **OFF** (falls back to advise-only) on any of:
- **repeated failure** — `failure_threshold` consecutive non-recoveries;
- **flap** — the same fix re-triggering `flap_threshold` times in `flap_window_s`;
- **rate cap** — more than `max_actions_per_window` applies per `window_s`.

Tripped → `OPEN` (advise-only) → after `breaker_cooldown_s` → `HALF_OPEN` (one
trial) → success closes it, failure re-opens. Reset manually via
`CircuitBreaker.reset()`.

## Scale: per-GPU + fleet blast radius

Each GPU gets its **own** `RemediationManager` (keyed `node_id="host:gpuN"`), so
one card verifying a fix never freezes the others and each has an independent
breaker + approval queue. A shared **`FleetCoordinator`** caps how many GPUs may
be actively remediating at once across the box/fleet — the fleet-wide blast
radius — and the operating mode is one box-wide kill-switch. This is what lets
the same layer serve one llama.cpp box and a thousand-GPU cluster.

## The global kill-switch (operating mode)

One setting forces the whole layer's behavior:

| mode | behavior |
|---|---|
| `off` | layer does nothing |
| `advise` | recommend fixes only; never touch the box *(safe default)* |
| `dry_run` | build & log the exact real commands, but never execute them |
| `auto` | unattended auto-apply for non-disruptive fixes (disruptive still needs approval) |

Flip it three ways, any time:
- **setup wizard** (first run): `et-remediation setup`
- **CLI**: `et-remediation mode advise`
- **dashboard / API**: `POST /api/remediation/mode {"mode":"advise"}`

Until setup is completed on a fresh box, the layer stays in **advise** — it never
actuates blind.

### How to disable auto-apply for ops

```bash
et-remediation mode advise     # or: off
# or, against a running monitor:
curl -XPOST localhost:7070/api/remediation/mode -d '{"mode":"advise"}'
```

## Backends (one interface, three implementations)

`Actuator` (capabilities · snapshot_state · apply · rollback):
- **`DataCenterActuator`** — `nvidia-smi`/NVML/DCGM (power limit, clock lock/reset),
  MPS/MIG, `renice`/`taskset`, orphan-kill; K8s/Slurm drain for the disruptive path.
- **`LlamaCppActuator`** — request-drain via the live metric, then restart
  `llama-server` with tuned flags (`-ngl`/`-t`/`-b`/`--cache-type-k|v`/`--mlock`).
- **`FakeActuator`** — records calls + drives a `FakeTelemetryModel`, so the whole
  loop is testable without a GPU.

Real commands execute **only** when `mode=auto` (and the request isn't dry-run):
the `CommandRunner` is the single chokepoint between "build the command" and "run
it".

## Strategies (root cause → action)

| Root cause | Action | Class |
|---|---|---|
| thermal/power throttle | raise power limit | non-disruptive |
| idle/zombie holding GPU | kill orphan PID (never the workload) | non-disruptive |
| CPU-bound preprocessing | renice loader workers | non-disruptive |
| memory fragmentation | free stale cached allocation | non-disruptive |
| data-pipeline starvation | restart w/ more workers | disruptive |
| distributed comm stall (NCCL) | restart w/ tuned NCCL env | disruptive |
| suboptimal runtime flags | restart llama-server w/ flags | disruptive |

Adding a strategy is adding one `ActionSpec` in `strategies.py`; the guardrails
are untouched.

## Dev

```bash
cd packages/remediation
uv run --extra dev pytest -q     # no GPU required (FakeActuator + sim harness)
```
