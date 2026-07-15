# Client optimization: RTX PRO 4000 Blackwell SFF + llama.cpp

Target box: **NVIDIA RTX PRO 4000 Blackwell SFF Edition** — 24 GB GDDR7, **432 GB/s**
(192-bit, 18 Gbps), 70 W, PCIe 5.0 x8. Single GPU, single `llama-server`, one
serving stream. Model + quant are **immutable**; every default here is
**output-lossless** (nothing that can change tokens is auto-applied).

## What ET got wrong about this box, and what changed

| Area | Was | Now |
|---|---|---|
| **GPU bandwidth** | SFF silently resolved to **672 GB/s** (matched `pro 4000 blackwell`) — 1.55× too high | Explicit `pro 4000 blackwell sff → 432`; full Blackwell RTX PRO table verified vs NVIDIA + TechPowerUp; longest-substring disambiguation (SFF vs full-size, Server vs Workstation 6000) |
| **Decode roofline** | `bandwidth / full-GGUF-bytes` — for an MoE this overstates bytes/token ~10× → MBU > 1, absurd ceilings | MoE-aware: reads `<arch>.expert_count`/`expert_used_count`, sums routed-expert (`*_exps`) vs always-active tensor bytes via data-offset deltas → **active-bytes-per-token** ceiling. Dense unchanged; any unreadable tensor block degrades to full-weight |
| **Prefill vs decode** | End-to-end tok/s could read as "decode" | Decode tok/s kept strictly separate; TTFT + prefill tok/s surfaced; new **PREFILL_BOUND** verdict fires on cold long contexts (high TTFT + healthy decode) and recommends prefix caching, never a decode lever |
| **Prefix cache** | Relied on removed `/metrics` KV counters | Reads `/slots` (`n_prompt_tokens`, `n_prompt_tokens_cache`) → prefix-cache **hit rate**; TTFT derived from uncached tokens ÷ observed prefill rate |
| **KV VRAM budget** | none | Reads GQA attention keys from GGUF → KV bytes/token; `WorkloadSpec.vram_fit(ctx)` reports weights+KV vs the 24 GB budget (predictive saturation), and the actuator **refuses a flag set that would OOM** before proposing it |
| **KV-cache quant** | `--cache-type-k/v q8_0` in the **default** restart flags (quality-affecting!) | **Opt-in only** (`kv_quant`/explicit knob); never in the lossless default set |
| **Prefill/TTFT lever** | none | `--cache-reuse 256` (output-lossless prefix-KV reuse) on by default; `-b` prefill-batch knob |
| **70 W thermals** | Power-limit fix hard-assumed a **300 W** base → proposed ~345 W on any card | Sizes the raise off the card's **real** current limit, clamps to its known max, and **refuses (advise cooling) when a card is already at its cap** — a 70 W SFF never gets a nonsense over-cap limit |

All restarts remain **DISRUPTIVE → human-gated** (drain + approval). Speculative
decoding stays **advisory** (output-identical but needs a draft-model artifact).

## Expected TPS impact per client scenario (physics-honest)

420-token generations. e2e = tokens ÷ wall; wall = TTFT + generation.

| Scenario | Now (e2e) | Lever | Expected |
|---|---|---|---|
| Short chat | 92.8 | none needed (at the wall, healthy) | **maintained ≥80**, not regressed |
| 8K cold | 50.5 (TTFT 3.3 s) | `--cache-reuse` + prefill batch cut cold TTFT; warm on repeat | ~65–72 cold; **~80 warm** |
| 30K warm | 76.1 (TTFT 0.31 s) | maximize prefix-cache hit rate; full offload + flash-attn | **~76–80** (essentially at target) |
| 30K cold | 21.1 (TTFT 14 s) | make cold **rare** via prefix reuse (14 s → 0.31 s warm) | cold ~27; the real fix is turning cold into warm |

### What is arithmetically impossible (do not chase)
**30K-cold ≥ 80 tok/s e2e cannot happen.** At 80 e2e, 420 tokens must complete in
5.25 s of wall *total*. Decode alone at 30K context (~73 tok/s, KV read grows with
context) takes 420 ÷ 73 ≈ **5.75 s** — already over budget **before any prefill**.
So even with TTFT = 0 the ceiling at 30K is ~73 e2e. The honest goal at 30K is to
make requests **warm** (prefix cache), where e2e is ~76–80 and decode, not prefill,
is the (physical) limit. We do not degrade quality to fake this number.

## Verification
539 tests pass (remediation 142, monitor 137, engine 140, agent 111, web/api 9);
ruff + mypy clean on touched packages; adversarial pass (missing/garbage GGUF,
unknown GPU, no-bandwidth spec, partial offload, MoE-fallback) all degrade
gracefully with no exceptions. Four client scenarios are encoded as analyzer
tests asserting the correct verdict for each.

## Scope delivered
All four phases are implemented and tested:
- **Phase 1** — SFF bandwidth fix, MoE-aware roofline, prefill-bound verdict.
- **Phase 2** — prefix-cache observability from `/slots` (2a) **and** KV-cache
  VRAM budget from GGUF with predictive saturation (2b).
- **Phase 3** — quality-lossless tuned config (opt-in KV-quant, `--cache-reuse`,
  prefill batch), 70 W thermal sanity, and a VRAM-budget validator that refuses
  an OOM flag set before it can be proposed.
- **Phase 4** — full cross-package verification + this report.

Remaining as advisory-only by design (never auto-applied): speculative decoding
(needs a draft-model artifact) and KV-cache quantization (quality-affecting).

## Open questions for the client
1. **Exact model + quant** (name + GGUF)? Physics say it's an MoE with ~3B active,
   not a dense 27B — ET now reads this from GGUF at runtime, but confirm.
2. Is the **80 tok/s target decode or end-to-end**? They are different problems;
   30K-cold e2e ≥80 is impossible (above), warm is reachable.
3. What is the **prefix-reuse pattern** of their traffic (shared system prompts /
   repeated long contexts)? That sets how often cold 30K prefills actually occur,
   which is the single biggest driver of their effective tok/s.
