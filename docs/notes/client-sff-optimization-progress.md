# Client SFF Optimization — Progress / Handoff

**Branch:** `client-sff-optimization`
**Status:** ✅ COMPLETE — all 4 phases done, verified, pushed (538 tests passing).
The comprehensive hand-off record is in the repo-root **`CLAUDE.md`** (auto-loaded each
session); this file is the supplementary per-phase log and must not contradict it.
Final write-up: `client-sff-optimization-summary.md`.

This is an autonomous multi-phase task. Commit + **push** after every logical unit
(the user explicitly wants pushes so they can continue across devices — this overrides
the original prompt's "don't push").

---

## Mission (verbatim intent)

Optimize ET (this repo) for one specific client deployment and maximize the client's
effective tokens/sec **without changing their model or quantization** and **without
degrading output quality**. Leave the repo CI-green with all changes tested. Work
through the phases below; if a phase reveals the plan is wrong, adapt and document.

### Client system profile (ground truth)
- **GPU:** NVIDIA RTX PRO 4000 Blackwell **SFF** Edition — 24 GB GDDR7, **432 GB/s**
  memory bandwidth (192-bit, 18 Gbps), 70 W TGP, PCIe 5.0 x8, 8960 CUDA cores.
  NOT the full-size RTX PRO 4000 Blackwell (672 GB/s). 70 W chassis → throttle risk.
- **Serving stack:** llama.cpp `llama-server`, single GPU, single box.
- **Model:** client says "Qwen dense 27B", but benchmark physics contradict dense
  (a dense 27B on 432 GB/s caps ~27 tok/s single-stream decode at 100% MBU; client
  measures up to ~94). Numbers are only consistent with a **MoE ~3B active** (likely
  Qwen3-30B-A3B family). ET must decide architecture from GGUF metadata at runtime.
  **Model + quant are immutable.**

### Client benchmark (all scenarios generated ~420 output tokens; reported "tok/s" is
END-TO-END = tokens ÷ wall, NOT pure decode):

| Scenario           | TTFT   | Reported tok/s | Wall   | Derived pure decode |
|--------------------|--------|----------------|--------|---------------------|
| Short chat         | 0.07 s | 92.8           | 4.6 s  | ~94 tok/s           |
| 8K ctx, cold cache | 3.3 s  | 50.5           | 8.2 s  | ~85 tok/s           |
| 30K ctx, cold cache| 14.0 s | 21.1           | 19.7 s | ~73 tok/s           |
| 30K ctx, warm cache| 0.31 s | 76.1           | 5.6 s  | ~80 tok/s           |

### Diagnosis (established, drives the whole task)
Decode is healthy (~73–94 tok/s across contexts — consistent with MoE ~3B active).
The system is **prefill-bound on cold long contexts**. Warm prefix cache takes 30K-cold
from 14.0 s → 0.31 s TTFT. **Levers = prefix-cache hit rate + prefill speed, NOT decode.**

### Targets (be physics-honest)
- ≥80 tok/s end-to-end for short-chat + warm-cache (already 92.8 / 76.1 — close the
  76.1 gap, never regress). Minimize cold TTFT (8K 3.3 s, 30K 14.0 s). Max cache hit rate.
- **30K-cold ≥80 e2e is arithmetically impossible** (420 tok at 80 e2e = 5.25 s wall;
  decode alone eats ~5.25 s). Do not chase it; do not hack quality to fake it.

### Hard constraints
1. Model + quant fixed. 2. Quality must not degrade — anything output-altering
(KV-cache quant `--cache-type-k/v`, reduced ctx, sampling) is opt-in advisory only,
never default, never auto-applied. 3. Never weaken the remediation safety model
(non-disruptive vs disruptive, human gating, protected-PID, breaker, kill switch;
any llama-server restart = DISRUPTIVE/human-gated + checkpoint + drain). 4. All 485+
existing tests keep passing; new tests per change; ruff + mypy clean in touched
packages. 5. Verify any new hardware spec via web before hardcoding.

---

## Key repo facts discovered (so a resumer doesn't re-derive)
- Roofline lives in `packages/monitor/src/et_monitor/perf.py`:
  `_BANDWIDTH_GB_S` table (has WRONG `"pro 4000 blackwell": 672` for the SFF — real 432),
  `read_gguf_metadata()`, `WorkloadSpec`, `roofline()`. Roofline currently divides
  bandwidth by FULL GGUF bytes → wrong for MoE (should be active-expert bytes/token).
- Verdict engine: `packages/monitor/src/et_monitor/analyzer.py::analyze()`. Verdicts in
  `types.py::Verdict`. llama.cpp `predicted_tokens_seconds` (decode) vs
  `prompt_tokens_seconds` (prefill) are ALREADY separate; snapshot carries
  `gen_tokens_per_s` (from predicted_* counter delta) and `prompt_tokens_per_s`.
  So pure decode is already separated from prefill in `state.py::tick()`.
- llama metrics scraper: `llama.py` (`/metrics` Prometheus + `/props`). No `/slots`
  endpoint scraping yet (Phase 2 needs it for prefix-cache hit rate).
- Remediation actuator: `packages/remediation/src/et_remediation/actuators/llamacpp.py`.
- Tests: `packages/monitor/tests/` (test_perf.py, test_roofline_analyzer.py,
  test_analyzer.py, test_llama_parser.py, ...). Run per package with `uv run pytest`.

---

## Phase status

- [x] **Phase 1.1** — DONE. `perf.py` `_BANDWIDTH_GB_S`: added `pro 4000 blackwell sff`
      =432 (wins longest-substring over full `pro 4000 blackwell`=672) and
      `pro 2000 blackwell`=288. Verified vs NVIDIA pages: 6000=1792, 5000=1344, 4500=896,
      4000-full=672, 4000-SFF=432, 2000=288. Fixed the test that hard-coded SFF==672;
      added disambiguation + lineup tests. `tests/test_perf.py`+`test_detect.py` green (18).
- [x] **Phase 1.2** — DONE. GGUF reader now captures expert_count/expert_used_count/
      embedding_length/expert_feed_forward_length (verified key names vs llama.cpp
      gguf-py); stops at first `tokenizer.` key (still skips vocab arrays).
      `estimate_moe_active_bytes()` scales full weight bytes by active-param fraction
      (routed-expert geometry + param_count; coarse expert-ratio fallback; clamped
      >= expert ratio; never raises). `WorkloadSpec.active_bytes` + `per_token_bytes()`
      + `is_moe`; roofline divides by active bytes (NOT full GGUF) so MoE MBU/ceiling
      are right (~10x). Wired into detect.py + state workload dict. `model_bytes` still
      = full VRAM footprint (offload/fit unchanged). Tests: dense unchanged, MoE
      geometry (~A3B), coarse fallback, missing-used-count->dense, clamp, tokenizer-stop,
      per_token_bytes, roofline-uses-active (incl. the client 94 tok/s => ~0.4 MBU on
      MoE vs >3x-over-ceiling on wrong-dense). Monitor suite 122 green, ruff+mypy clean.
- [x] **Phase 1.3** — DONE. `gen_tokens_per_s` was already pure decode (predicted_*),
      `prompt_tokens_per_s` pure prefill — documented that invariant in types.py. Added
      Snapshot.prompt_tokens_total / predicted_tokens_total / ttft_s (populated in
      state.py + serialized). Analyzer computes `prefill_fraction` = share of serving
      time spent prefilling (counter deltas / representative throughputs) and surfaces
      prefill_tokens_per_s / prefill_fraction / ttft_s in metrics. New Verdict.PREFILL_BOUND
      (Rule 3.6, before decode): fires when prefill_fraction >= 0.5 + decode healthy +
      active; recommends prefix caching, explicitly says NOT to chase decode. Tests: all
      four client scenarios (30K-cold->PREFILL_BOUND, short/warm->HEALTHY, 8K-cold<0.5)
      + decode-rate-is-not-end-to-end. Monitor suite 127 green; ruff+mypy clean.
      NOTE: ttft_s currently always None (Phase 2 /slots will populate it precisely);
      the verdict uses prefill_fraction, which is available now.
- [~] **Phase 2** — MOSTLY DONE (scoped to what llama.cpp actually exposes). KEY
      FINDING: llama.cpp exposes **no direct prefix-cache-hit counter** (verified vs
      server README) — do NOT chase one. It DOES expose exact prefill/decode wall-time
      (`prompt_seconds_total`, `tokens_predicted_seconds_total`); added those +
      `n_busy_slots_per_decode` to the scraper/snapshot. `_prefill_fraction` now uses the
      exact time counters (throughput estimate is fallback) — warm vs cold cache is
      visible as prefill-time collapse, which IS the cache-effectiveness signal, and
      feeds PREFILL_BOUND (the repeatedly-cold-long-context verdict). Added
      `_kv_headroom_gb` (VRAM left for KV after full resident weights, using full
      footprint not MoE active) -> `kv_headroom_gb` metric; KV_CACHE_PRESSURE advice is
      now headroom-aware (won't say "raise --ctx-size" when it would OOM) and labels
      KV-quant as output-altering/opt-in (quality constraint). Monitor suite 130 green,
      ruff+mypy clean.
      REMAINING (optional, lower value): a best-effort `/slots` reader for per-slot
      `n_past` (behind `--slots`, privacy-sensitive, usually off) to count warm slots;
      "tokens/TTFT saved" is not derivable without a cache-hit counter. Documented in
      llama.py. Decide in Phase 4 whether to add the /slots reader or leave as-is.
- [x] **Phase 3** — DONE. Found the remediation layer already had tuned-restart /
      offload-fix / spec-decode strategies with VRAM guards. Added what was missing for
      THIS client: (1) a **prefix-cache remediation** for the new PREFILL_BOUND verdict —
      RootCause.COLD_PREFIX_CACHE, `_build_prefill_cache` (sets `--cache-reuse`, keeps
      model resident, flash-attn, ubatch on headroom; OUTPUT-LOSSLESS — no KV-quant/model
      change), ActionSpec PREFILL_COLD_CACHE (DISRUPTIVE/human-gated/drained), mapped
      `"prefill_bound"`, added `--cache-reuse` to the actuator flag map. Verified on a new
      `prefill_relieved` predicate (prefill_fraction drop) via a new WindowSummary
      `prefill_fraction` computed from the Phase-2 time counters. (2) **KV-quant made
      opt-in** (`allow_kv_quant` knob) in `_build_restart_llama` — was auto-added by
      default, which violated the quality constraint. Existing VRAM guards (-ngl not
      grown on full card, draft-model OOM refusal, model-fits check) already satisfy
      "validate VRAM budget"; spec-decode already advisory (needs draft_model knob).
      Remediation suite 139 green, monitor 130 green, ruff+mypy clean. Safety model
      untouched (disruptive still APPROVAL_REQUIRED in AUTO).
      NOTE: 70 W throttle thresholds — the SET_POWER_LIMIT builder defaults to +15%
      headroom off the READ current limit (not a hardcoded 300 W assumption), so it
      scales to a 70 W card. Verify explicitly in Phase 4.
- [x] **Phase 4** — DONE. All five suites green: **engine 140, agent 111, monitor 139,
      remediation 139, web/api 9 = 538 passed, 0 failed**; ruff clean everywhere; mypy
      clean on all touched files. Added `test_client_sff_scenario.py` (end-to-end client
      numbers + adversarial: unknown GPU, MoE-missing-used-count->dense, partial MoE
      offload, dense unchanged). Written summary: `client-sff-optimization-summary.md`.
      One verified open item (70 W power-limit can propose above-cap; driver rejects
      safely) — documented, deliberately not fixed (needs sampler max-power plumbing in
      the safety-critical path). TASK COMPLETE.

## Decision / change log (append newest at bottom)
- Set up branch + handoff doc + CLAUDE.md resume pointer. Pushing enabled per user.
