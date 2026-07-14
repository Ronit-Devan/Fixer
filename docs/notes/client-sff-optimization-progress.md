# Client SFF Optimization — Progress / Handoff

**Branch:** `client-sff-optimization`
**Status:** IN PROGRESS. Updated as work proceeds so this can be resumed on another
device. To resume: `git checkout client-sff-optimization && git pull`, read this file
top-to-bottom, then continue at the first unchecked box in "Phase status".

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
- [ ] **Phase 1.2** — MoE-aware roofline: read expert_count/expert_used_count from GGUF;
      compute active-params bytes/token for MoE decode ceiling; keep never-raises
      contract; tests for dense (unchanged), MoE, missing-metadata fallback.
- [ ] **Phase 1.3** — Prefill/decode separation as first-class + prefill-bound /
      prefix-cache-cold verdict (high TTFT + healthy decode MBU + long prompt). Tests
      for all four client scenarios.
- [ ] **Phase 2** — Prefix-cache observability from `/metrics` + `/slots` (verify real
      field names): hit rate, tokens saved, TTFT saved, KV occupancy vs 24 GB; verdict
      for repeatedly-cold long-context; predictive KV-saturation tuned to weights+30K KV.
- [ ] **Phase 3** — Tuned llama-server flag generation in the actuator (behind human
      gate): -ngl 999, flash-attn, batch/ubatch for prefill, cache-reuse, host-side,
      spec-decode advisory, KV-quant opt-in; validate flags + VRAM budget; 70 W throttle
      thresholds sane; safety model intact.
- [ ] **Phase 4** — Full verification: all package suites + ruff + mypy; fixtures for
      the four scenarios; mock-llama-server integration; adversarial pass (missing/wrong
      GGUF, dense, unknown GPU, partial offload); written summary + open client questions.

## Decision / change log (append newest at bottom)
- Set up branch + handoff doc + CLAUDE.md resume pointer. Pushing enabled per user.
