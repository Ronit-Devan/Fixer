# Client SFF Optimization — Final Summary

**Branch:** `client-sff-optimization` (all work committed + pushed).
**Client:** RTX PRO 4000 Blackwell **SFF** (24 GB, 432 GB/s, 70 W), llama.cpp
`llama-server`, single GPU, serving a Qwen **MoE** (~3B active) at a fixed quant.
**Goal:** maximize effective tok/s without changing the model/quant or hurting quality.

## Headline

ET was quietly **wrong about this exact box** in three ways that all pushed the
same direction — it would have told the client to chase decode speed they don't
need. The root cause of the client's low reported tok/s is **cold prefill (TTFT)
on long contexts, not decode**. Every change below makes ET report that correctly
and gives the operator the lever that actually helps (prefix caching), all while
respecting the fixed model/quant and the quality + safety constraints.

Test totals across the repo: **engine 140, agent 111, monitor 139, remediation
139, web/api 9 = 528 passing**; ruff + mypy clean on every touched package. (The
monitor count reflects removing one bug-encoding test and adding ~30 new ones.)

## What changed, by phase

**1.1 — SFF bandwidth (correctness).** The bandwidth table mapped
`pro 4000 blackwell` → 672 GB/s; the SFF's NVML name contains that substring, so
it silently inherited **672 instead of its real 432** (the 70 W SFF downclocks its
GDDR7 — verified against NVIDIA's product page + datasheet). That made every
single-stream ceiling ~1.55× too high and would mislabel a card *at* its wall as
"fixable." Added an explicit longer SFF key (432) + `pro 2000 blackwell` (288);
audited the whole Blackwell workstation line. `perf.py`, `test_perf.py`.

**1.2 — MoE-aware roofline (correctness).** The roofline divided bandwidth by the
**full** GGUF bytes. For an MoE only the active experts stream per token (~1/10th
the bytes), so ET reported an MBU ~10× too low and a ceiling ~10× too low — for
this client, the difference between "impossible, >3× over a dense-27B ceiling" and
"healthy ~0.4 MBU on a ~3B-active MoE." The GGUF reader now reads
`expert_count`/`expert_used_count`/`embedding_length`/`expert_feed_forward_length`
(key names verified vs llama.cpp gguf-py) and `estimate_moe_active_bytes` computes
the active bytes/token; the roofline divides by that. `model_bytes` stays the full
VRAM footprint (offload/fit logic untouched). `perf.py`, `detect.py`, `state.py`.

**1.3 — Prefill vs decode + PREFILL_BOUND verdict (the core insight).**
`gen_tokens_per_s` was already pure decode; made the invariant explicit and added
the cumulative token/time counters. New `PREFILL_BOUND` verdict fires when most
serving *time* is prefill but decode is healthy — and it explicitly tells the
operator **not** to chase decode tok/s, pointing at prefix caching instead. Tests
reproduce all four client scenarios. `types.py`, `state.py`, `analyzer.py`.

**2 — Exact timing + KV budget.** Verified llama.cpp exposes **no** prefix-cache-hit
counter, so cache effectiveness is observed honestly via the exact
`prompt_seconds_total` / `tokens_predicted_seconds_total` counters (a warm cache
shows as prefill-time collapse). `prefill_fraction` is now *measured*, not
estimated. Added `kv_headroom_gb` (VRAM left for KV after full resident weights),
which makes KV-pressure advice headroom-aware (won't tell a full 24 GB box to raise
`--ctx-size` and OOM) and labels KV-quant as output-altering/opt-in.

**3 — Prefix-cache remediation + KV-quant opt-in.** New `COLD_PREFIX_CACHE` root
cause and `PREFILL_COLD_CACHE` strategy: a human-gated, drained restart with
`--cache-reuse` (output-lossless prefix caching) — the client's actual fix. It
verifies on **prefill relief** (`prefill_fraction` dropping), not decode tok/s, so
a genuine fix is confirmed and a no-op rolls back. Also made KV-cache quantization
**opt-in** (`allow_kv_quant`) instead of auto-applied, per the quality constraint.
Safety model untouched — disruptive restarts still require approval even in AUTO.

**4 — Verification.** All five suites green; new end-to-end client-scenario +
adversarial regression file; ruff + mypy clean.

## Expected impact per benchmark scenario

Reported "tok/s" was end-to-end (tokens ÷ wall). Pure decode is ~73–94 tok/s
throughout — already at/above 80. The gap is prefill on cold long contexts.

| Scenario | Reported e2e | Bottleneck | ET verdict now | Lever |
|---|---|---|---|---|
| Short chat | 92.8 | none | HEALTHY / decode | — (already >80) |
| 8K cold | 50.5 | prefill (40%) | decode/near-threshold | prefix cache / bigger ubatch |
| 30K cold | 21.1 | **prefill (71%)** | **PREFILL_BOUND** | **`--cache-reuse` (proven: → 76.1)** |
| 30K warm | 76.1 | none | HEALTHY | keep cache warm |

**Physics-honest ceiling:** 30K-cold at **≥80 tok/s end-to-end is impossible** — 420
tokens at 80 e2e = 5.25 s wall, but decode alone (~80 tok/s) already needs ~5.25 s,
leaving zero for the unavoidable cold prefill of a novel 30K prompt. ET now says so
instead of implying a decode fix. The reachable win is making cold long-context
prefills *rare* (prefix-cache hit rate) and *faster* (prefill batch), which the
warm row already demonstrates (14.0 s → 0.31 s TTFT).

## Open items (verified, deliberately not changed)

1. **70 W power-limit remediation.** `_build_power_limit` raises the *current*
   enforced limit by 15%. On a 70 W card already at its cap it would propose ~81 W;
   the driver safely rejects it and the circuit breaker trips (no harm), but it's a
   pointless proposal. Proper fix = plumb the card's max power limit
   (`nvmlDeviceGetPowerManagementLimitConstraints`) through the sampler and clamp —
   a change in the safety-critical actuation path, left as a follow-up. For an SFF
   card, throttle is usually heat, not a power cap; the monitor already advises
   cooling for that case.
2. **Prefix-cache hit rate (%)** is not derivable — llama.cpp exposes no hit
   counter. A best-effort `/slots` `n_past` reader (behind `--slots`, usually off)
   could count warm slots; not built.

## Questions for the client

1. **Exact model + quant?** Strongly evidenced as a ~3B-active MoE (likely
   Qwen3-30B-A3B), *not* a dense 27B — the measured 92.8 tok/s is physically
   impossible for a dense 27B on 432 GB/s. Confirm so the active-bytes estimate can
   be pinned (or overridden exactly).
2. **Is the 80 tok/s target per-stream (decode) or end-to-end?** Decode already
   clears it; end-to-end clears it except cold long-context, which is a prefill/
   cache problem, not decode.
3. **Traffic's prefix-reuse pattern?** Shared system prompt / RAG preamble across
   requests → `--cache-reuse` is a large win. Mostly unique long prompts → the cold
   prefill is inherent and the lever is prefill batch + accepting the TTFT floor.
