# CLAUDE.md

Repo-context + hand-off file (auto-loaded each session). The lower half is the
complete record of the **client SFF optimization** task on branch
`client-sff-optimization`.

## Repo orientation

ET attributes GPU idleness to a root cause. Packages under `packages/`:
- `engine` — diagnoses a PyTorch Profiler trace (training-side). CLI `gpu-doctor`.
- `monitor` — live llama.cpp serving monitor + decode **roofline** (MBU / single-stream
  tok/s ceiling / partial-offload). Most client-optimization work lives here.
- `remediation` — safe actuation layer (auto-applies non-disruptive fixes, human-gates
  disruptive ones; never kills the running workload).
- `agent` — eBPF/torch live-capture agent. `web/api` — FastAPI backend; `web/app` — frontend.

Each package uses `uv`. Test a package: `cd packages/<pkg> && uv run pytest`
(web backend lives in `packages/web/api`). Lint: `uv run ruff check src/ tests/`.
Type-check (mypy is not a dep of every package): `uv run --with mypy mypy src/ --ignore-missing-imports`.
CI (`.github/workflows/ci.yml`) currently gates only the `engine` package and runs
mypy non-blocking (`|| true`).

---

# TASK: Client SFF optimization — ✅ COMPLETE

**Branch:** `client-sff-optimization` (8 commits, all pushed to
`origin` = github.com/Ronit-Devan/Fixer). Open a PR:
https://github.com/Ronit-Devan/Fixer/pull/new/client-sff-optimization

**Status:** All 4 phases done, verified, pushed. Full repo green:
**engine 140, agent 111, monitor 139, remediation 139, web/api 9 = 538 passing, 0 failing.**
ruff clean everywhere; mypy clean on all touched files. Remediation safety model untouched.

Finer detail lives in `docs/notes/client-sff-optimization-progress.md` (per-phase log)
and `docs/notes/client-sff-optimization-summary.md` (write-up + client questions). This
file is the standalone hand-off; those two are supplementary and must not contradict it.

## End goal (why this task exists)

Optimize ET for one client deployment and maximize their **effective tok/s** WITHOUT
changing the model or quantization and WITHOUT degrading output quality. The deliverable
is ET being *correct and useful about this exact box*, plus a remediation for the client's
real bottleneck — not a generic speedup.

**Client system:** NVIDIA RTX PRO 4000 Blackwell **SFF** (24 GB GDDR7, **432 GB/s**, 70 W,
PCIe 5.0 x8), llama.cpp `llama-server`, single GPU. Model claimed "Qwen dense 27B" but the
benchmark physics prove a **Mixture-of-Experts (~3B active**, likely Qwen3-30B-A3B) — a
dense 27B on 432 GB/s caps at ~27 tok/s single-stream, yet the client measures up to 94.

**Client benchmark** (all scenarios generated ~420 output tokens; the reported "tok/s" is
END-TO-END = tokens ÷ wall, NOT pure decode):

| Scenario | TTFT | Reported tok/s | Wall | Pure decode |
|---|---|---|---|---|
| Short chat | 0.07 s | 92.8 | 4.6 s | ~94 tok/s |
| 8K cold | 3.3 s | 50.5 | 8.2 s | ~85 tok/s |
| 30K cold | 14.0 s | 21.1 | 19.7 s | ~73 tok/s |
| 30K warm | 0.31 s | 76.1 | 5.6 s | ~80 tok/s |

**The core finding:** decode is healthy (~73–94 tok/s throughout). The low reported tok/s
on cold long contexts is **prefill (TTFT), not decode**. The lever is prefix caching, which
the warm row proves (14.0 s → 0.31 s TTFT). ET was silently wrong about this box in three
compounding ways — all now fixed — that would have sent the client chasing decode speed.

**Physics-honest ceiling (do not "fix" this):** 30K-cold at ≥80 tok/s **end-to-end is
impossible** — 420 tokens at 80 e2e = 5.25 s wall, but decode alone (~80 tok/s) already
needs ~5.25 s, leaving zero for the novel prompt's cold prefill. ET now says so.

## Hard constraints (were honored)

1. Model + quant are FIXED. 2. Quality must not degrade — anything output-altering
(KV-cache quant, reduced ctx, sampling) is opt-in advisory only, never default/auto.
3. Never weaken the remediation safety model (non-disruptive vs disruptive, human gating,
protected-PID, breaker, kill switch; llama-server restart = DISRUPTIVE → human-gated +
drained). 4. All tests pass, new tests per change, ruff + mypy clean. 5. Verify hardware
specs on the web before hardcoding.

## What changed — complete edit list

**Commit `21a0b50` — Phase 1.1: SFF bandwidth (432, not 672).**
The `_BANDWIDTH_GB_S` table mapped `pro 4000 blackwell`→672; the SFF's NVML name contains
that substring so it inherited 672 (real 432 — the 70 W SFF downclocks GDDR7; verified vs
NVIDIA product page + datasheet). → ~1.55× too-high ceilings, mislabeling a card at its wall.
- `packages/monitor/src/et_monitor/perf.py`: added `pro 4000 blackwell sff`=432 (wins
  longest-substring) + `pro 2000 blackwell`=288; audited the Blackwell line (6000=1792,
  5000=1344, 4500=896, 4000-full=672).
- `packages/monitor/tests/test_perf.py`: removed the test that hard-coded SFF==672; added
  SFF-vs-full disambiguation + lineup tests.

**Commit `32de5ea` — Phase 1.2: MoE-aware roofline.**
The roofline divided bandwidth by the FULL GGUF bytes; an MoE streams only active experts
(~1/10th) per token → MBU/ceiling ~10× wrong. `model_bytes` stays the full VRAM footprint;
a new `active_bytes` is the roofline denominator.
- `perf.py`: `GgufInfo` gains `expert_count`/`expert_used_count`/`embedding_length`/
  `expert_ffn_len` + `is_moe`; `read_gguf_metadata` reads them (key names verified vs
  llama.cpp gguf-py) and now STOPS at the first `tokenizer.` key (still skips vocab arrays);
  new `estimate_moe_active_bytes()` (full bytes × active-param fraction from routed-expert
  geometry + param_count; coarse expert-ratio fallback; clamped ≥ expert ratio; never
  raises); `WorkloadSpec.active_bytes` + `per_token_bytes()` + `is_moe`; `roofline()` divides
  by `per_token_bytes()`.
- `detect.py`: computes/sets `active_bytes`, notes + preview line.
- `state.py`: `_workload_dict` surfaces `is_moe` + `active_gb`.
- `tests/test_perf.py`: dense-unchanged, MoE-geometry, coarse-fallback, missing-used-count,
  clamp, tokenizer-stop, per_token_bytes, roofline-uses-active (incl. client 94 tok/s).

**Commit `baabb80` — Phase 1.3: prefill/decode split + `PREFILL_BOUND` verdict.**
`gen_tokens_per_s` was already pure decode (predicted_* counter); made the invariant explicit.
- `types.py`: `Verdict.PREFILL_BOUND` + title; `Snapshot` gains `prompt_tokens_total`,
  `predicted_tokens_total`, `ttft_s`.
- `state.py`: populate + serialize the counters.
- `analyzer.py`: `Thresholds.prefill_bound_fraction=0.5`; helpers `_mean_positive`,
  `_counter_delta`, `_prefill_fraction`; prefill metrics surfaced; **Rule 3.6 PREFILL_BOUND**
  (fires when prefill dominates serving TIME but decode is healthy; explicitly says DON'T
  chase decode; recommends prefix caching). Ranked above the decode verdict.
- `tests/test_analyzer.py`: all four client scenarios + "decode rate ≠ end-to-end rate".

**Commit `4654f0e` — Phase 2: exact timing + KV budget.**
Verified llama.cpp exposes NO prefix-cache-hit counter; cache effectiveness = prefill-time
collapse via the exact `*_seconds_total` counters it DOES expose.
- `llama.py`: scrape `prompt_seconds_total`, `tokens_predicted_seconds_total`,
  `n_busy_slots_per_decode`; docstring documents the no-hit-counter finding.
- `types.py`/`state.py`: `Snapshot.prompt_seconds_total`/`predicted_seconds_total` + serialize.
- `analyzer.py`: `_prefill_fraction` now PREFERS the exact time counters (throughput estimate
  is fallback); new `_kv_headroom_gb` → `kv_headroom_gb` metric; **KV_CACHE_PRESSURE advice is
  headroom-aware** (won't tell a full 24 GB box to raise `--ctx-size` and OOM) and labels
  KV-quant as output-altering/opt-in.
- `tests/test_analyzer.py`: exact-time-counter path + kv-headroom (tight vs ample).

**Commit `460fa0b` — Phase 3: prefix-cache remediation + KV-quant opt-in.**
- `remediation/telemetry.py`: `_counter_delta`/`_prefill_fraction`; `WindowSummary.prefill_fraction`
  (from the Phase-2 time counters).
- `remediation/verify.py`: `prefill_relieved(pre, post)` recovery predicate.
- `remediation/rootcause.py`: `RootCause.COLD_PREFIX_CACHE`; map `"prefill_bound"` → it.
- `remediation/strategies.py`: `_build_prefill_cache` (restart with `--cache-reuse`, model
  kept resident, flash-attn, ubatch only on VRAM headroom; **output-lossless** — no KV-quant/
  model change); `PREFILL_COLD_CACHE` ActionSpec (DISRUPTIVE → approval-gated + drained;
  verified on `prefill_relieved`); registered. **KV-quant made OPT-IN** (`allow_kv_quant`
  knob) in `_build_restart_llama` — was auto-added by default (quality-constraint violation).
- `remediation/actuators/llamacpp.py`: `_FLAG_MAP` gains `cache_reuse` → `--cache-reuse`.
- `tests/test_llamacpp_tuning.py`: mapping, lossless builder, KV-quant opt-in, actuator
  render, `prefill_relieved`, end-to-end approval.

**Commits `c12bd3a`, `22d5979` — Phase 4: verification + docs.**
- `monitor/tests/test_client_sff_scenario.py` (new): end-to-end on the real client numbers
  (SFF=432, MoE ceiling realistic, four scenarios) + adversarial (unknown GPU → no roofline,
  MoE-missing-used-count → dense, partial MoE offload flagged, dense unchanged).
- `docs/notes/client-sff-optimization-summary.md` + progress doc.

## Open items (verified, deliberately NOT changed)

1. **70 W power-limit remediation** (`strategies._build_power_limit`): raises the *current*
   enforced limit by 15%. On a 70 W card already at its cap it proposes ~81 W; the driver
   safely rejects it and the circuit breaker trips (no harm), but it's a pointless proposal.
   Proper fix = plumb the card's max power limit
   (`nvmlDeviceGetPowerManagementLimitConstraints`) through `gpu.py` and clamp — a change in
   the safety-critical actuation path, left as a follow-up. For an SFF card, throttle is
   usually HEAT (raising power won't help); the monitor already advises cooling.
2. **Prefix-cache hit-rate %** is not derivable (no llama.cpp counter). A best-effort
   `/slots` `n_past` reader (behind `--slots`, usually off) could count warm slots; not built.

## Questions for the client (in the summary doc)

1. Exact model + quant? (Confirm the MoE — the active-bytes estimate can then be pinned or
   overridden exactly. It's inferred from physics; ET keys off runtime GGUF metadata, so it
   self-corrects from the real `expert_count`/`expert_used_count`.)
2. Is 80 tok/s per-stream (decode — already met) or end-to-end (met except cold long-context)?
3. Traffic's prefix-reuse pattern? (Shared preamble → `--cache-reuse` is a big win; mostly
   unique long prompts → cold prefill is inherent, lever is prefill batch + accepting the TTFT floor.)

## If extending this work

Commit + **push** per logical unit (the user wants cross-device continuity). Keep the
per-phase log and summary in `docs/notes/` in sync with this file. Verify a package with
`cd packages/<pkg> && uv run pytest -q` and `uv run ruff check src/ tests/`.
