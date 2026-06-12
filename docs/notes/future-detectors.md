# Future detector ideas (with research grounding)

## Marconi-style prefix cache hit-rate (LLM inference)
Paper: Marconi (MLSys '25, arXiv 2411.19379)
Insight: Prefix cache hit-rate is a primary signal for LLM inference cost.
For hybrid LLMs (Mamba, Jamba, Zamba), naive prefix caching breaks down.
Build a separate verdict: PREFIX_CACHE_MISS_HIGH, with admission-policy
recommendations.

When to add: after we have a customer running inference traces, not training.

## DataStates-style checkpoint overhead (training reliability)
Paper: DataStates-LLM (HPDC '24, arXiv 2406.10707)
Insight: Synchronous checkpointing eats 12-43% of train time. Detectable
in traces by long stalls during torch.save / checkpoint hooks.
Add CHECKPOINT_BOUND verdict; recommend lazy async checkpointing.

When to add: once a design partner mentions checkpoint pain.

## SDC detection (training reliability)
Paper: Understanding SDC in LLM Training (arXiv 2502.12340)
Insight: Silent data corruption detectable via gradient norm anomalies
and NaN propagation. Doesn't fit a single profiler trace; needs a callback.
This is a separate product, not a Profiler-trace detector.

When to add: as a sidecar library, not part of gpu-doctor.

## Speculative decoding tuning (LLM inference)
Paper: Decoding Speculative Decoding (arXiv 2402.01528)
Insight: Optimal draft model gamma value depends on workload.
Auto-tune by analyzing decoder traces.

When to add: same as Marconi, after inference customers exist.
