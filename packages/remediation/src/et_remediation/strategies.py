"""The concrete remediation strategies, each tagged disruptive / non-disruptive.

This is the pluggable content of the registry. Adding a strategy is adding an
``ActionSpec`` here; the manager and guardrails are untouched. The seven the
design calls for:

  NON-DISRUPTIVE (auto-appliable via the guarded path)
    thermal/power throttle        -> raise power limit            SET_POWER_LIMIT
    idle/zombie holding the GPU    -> kill ONLY the orphan PID     KILL_ORPHAN_PROCESS
    CPU-bound preprocessing        -> renice the loader workers    RENICE_PROCESS
    memory fragmentation           -> free stale cached allocation FREE_STALE_CACHE

  DISRUPTIVE (checkpoint + drain + human approval; never auto)
    data-pipeline starvation       -> restart w/ more workers      DRAIN_AND_RESTART_WORKLOAD
    distributed comm stall (NCCL)  -> restart w/ tuned NCCL env     DRAIN_AND_RESTART_WORKLOAD
    suboptimal runtime flags       -> restart llama-server w/ flags RESTART_LLAMA_SERVER
"""

from __future__ import annotations

from et_remediation.actions import (
    ActionClass,
    ActionContext,
    ActionKind,
    ActionSpec,
)
from et_remediation.registry import ActionRegistry
from et_remediation.rootcause import RootCause
from et_remediation.verify import (
    clock_recovered,
    memory_freed,
    throughput_recovered,
    util_recovered,
)


# -- small context readers ---------------------------------------------------


def _orphan_pid(ctx: ActionContext) -> int | None:
    v = ctx.metrics.get("orphan_pid", ctx.knobs.get("orphan_pid"))
    return int(v) if v is not None else None


def _stale_pid(ctx: ActionContext) -> int | None:
    v = ctx.metrics.get("stale_pid", ctx.knobs.get("stale_pid"))
    return int(v) if v is not None else None


def _worker_pid(ctx: ActionContext) -> int | None:
    pids = ctx.metrics.get("dataloader_worker_pids") or ctx.knobs.get("dataloader_worker_pids")
    if pids:
        return int(pids[0])
    v = ctx.knobs.get("worker_pid")
    return int(v) if v is not None else None


# -- param builders ----------------------------------------------------------


def _build_power_limit(ctx: ActionContext) -> dict:
    # Raise the enforced power limit toward the card's cap by a headroom factor.
    # Falls back to a sane default target if we can't read the current limit.
    cur = ctx.knobs.get("current_power_limit_w") or ctx.metrics.get("power_limit_w")
    target = ctx.knobs.get("target_power_w")
    if target is None:
        headroom = float(ctx.knobs.get("power_headroom_pct", 15.0)) / 100.0
        base = float(cur) if cur else 300.0
        cap = float(ctx.knobs.get("max_power_limit_w", base * (1 + headroom)))
        target = min(base * (1 + headroom), cap)
    prior = {"power_limit_w": float(cur)} if cur else {}
    return {"power_limit_w": int(round(float(target))), "prior": prior}


def _build_kill_orphan(ctx: ActionContext) -> dict:
    pid = _orphan_pid(ctx)
    if pid is None:
        raise KeyError("no orphan_pid available; cannot safely kill")
    return {"pid": pid}


def _build_renice(ctx: ActionContext) -> dict:
    pid = _worker_pid(ctx)
    if pid is None:
        raise KeyError("no dataloader worker pid available; cannot renice")
    nice = int(ctx.knobs.get("target_nice", -5))
    return {"pid": pid, "nice": nice, "prior": {"nice": int(ctx.knobs.get("prior_nice", 0))}}


def _build_free_stale(ctx: ActionContext) -> dict:
    pid = _stale_pid(ctx)
    if pid is None:
        raise KeyError("no stale_pid available; cannot free stale cache")
    return {"pid": pid}


def _build_restart_dataloader(ctx: ActionContext) -> dict:
    return {
        "restart_command": ctx.knobs.get("restart_command"),
        "job_id": ctx.job_id,
        "tuned": {
            "num_workers": ctx.knobs.get("num_workers", 8),
            "persistent_workers": True,
            "pin_memory": True,
        },
    }


def _build_restart_nccl(ctx: ActionContext) -> dict:
    return {
        "restart_command": ctx.knobs.get("restart_command"),
        "job_id": ctx.job_id,
        "env": {
            "NCCL_ALGO": ctx.knobs.get("nccl_algo", "Ring"),
            "NCCL_IB_DISABLE": ctx.knobs.get("nccl_ib_disable", "0"),
            "NCCL_P2P_LEVEL": ctx.knobs.get("nccl_p2p_level", "NVL"),
        },
    }


_QUANTIZED_KV = frozenset({"q8_0", "q4_0", "q4_1", "q5_0", "q5_1"})


def _has_demand(m: dict) -> bool:
    """Is there real concurrent demand to justify adding parallel slots?

    Adding ``--parallel`` on a single-user box wastes KV cache and can *hurt*
    single-stream latency, so we only raise it when requests are actually
    queueing or more than one is in flight."""
    deferred = m.get("max_requests_deferred") or 0
    conc = m.get("mean_concurrency", m.get("mean_requests_processing")) or 0
    return deferred >= 1 or conc > 1.05


def _build_restart_llama(ctx: ActionContext) -> dict:
    """Tuned llama-server restart params, derived from the live signal.

    Demand-gated and headroom-capped so the restart improves throughput without
    regressing a single-user box or risking OOM:
      * ``--parallel`` is only raised above current when there's concurrent demand;
      * ``-ngl`` defaults to the model's full layer count, but is NOT raised when
        VRAM is already near full (that would OOM on reload);
      * KV cache is only quantized when it isn't already.
    """
    m = ctx.metrics
    k = ctx.knobs
    params: dict = {
        "model": k.get("model"),
        "restart_command": k.get("restart_command"),
        "prior_argv": k.get("prior_argv", []),
        "drain_timeout_s": k.get("drain_timeout_s", 30.0),
    }

    # parallel: hold current unless there is genuine concurrent demand.
    if _has_demand(m):
        params["parallel"] = int(k.get("target_parallel", 4))
        params["cont_batching"] = True
    elif k.get("parallel") is not None:
        params["parallel"] = int(k["parallel"])

    # -ngl: full offload by default; never add layers into an almost-full card.
    mem_ratio = m.get("mem_used_ratio")
    if mem_ratio is not None and mem_ratio >= 0.90:
        if k.get("n_gpu_layers") is not None:
            params["n_gpu_layers"] = int(k["n_gpu_layers"])  # hold; don't grow
    else:
        params["n_gpu_layers"] = int(k.get("n_gpu_layers", k.get("model_n_layers", 999)))

    # KV cache: quantize only if it isn't already (avoid needless quality changes).
    if str(k.get("current_cache_type_k", "")).lower() not in _QUANTIZED_KV:
        params["cache_type_k"] = k.get("cache_type_k", "q8_0")
        params["cache_type_v"] = k.get("cache_type_v", "q8_0")

    if k.get("mlock", True):
        params["mlock"] = True

    # Flash attention: smaller KV footprint per token and a prerequisite for the
    # KV-cache quantization set above (q8_0 K/V needs flash-attn). Rendered with an
    # explicit value because current llama.cpp's -fa flag takes [on|off|auto].
    # Default ON unless the operator explicitly disabled it.
    fa = k.get("flash_attn", True)
    if fa is not False:
        params["flash_attn"] = "on" if fa is True else fa

    # Micro-batch size primarily speeds up PREFILL (prompt ingestion), not
    # single-stream decode, and a larger ubatch grows the compute buffers (more
    # VRAM). So only raise it when there is clear VRAM headroom; otherwise leave
    # llama.cpp's default. This keeps the restart from OOMing a near-full card.
    ubatch = k.get("ubatch_size")
    if ubatch is not None:
        params["ubatch_size"] = int(ubatch)
    elif mem_ratio is not None and mem_ratio < 0.80:
        params["ubatch_size"] = 1024

    return params


def _build_spec_decode(ctx: ActionContext) -> dict:
    """Enable speculative decoding. Raises (advise-only fallback) when it can't help.

    Spec decode runs a small draft model to propose N tokens, then verifies them
    in one forward pass of the main model. On a bandwidth-bound single-stream box
    at the physical decode ceiling this is the primary lever for pushing tok/s
    higher: the verification pass amortises the weight read across N tokens. The
    realised speedup is acceptance-rate dependent (typically ~1.3-2x for a
    well-matched draft of the same model family), NOT guaranteed — which is why
    the restart is verified on actual tok/s and rolled back if it doesn't help.

    Refuses (so the engine falls through to advise) when:
      * no ``draft_model`` is configured — nothing to draft with;
      * VRAM is already near full — the draft model AND its own KV cache need
        room, and a box at the single-stream wall already has the main model
        fully resident; loading a second model would OOM on reload.
    """
    m = ctx.metrics
    k = ctx.knobs
    draft_model = k.get("draft_model") or k.get("model_draft")
    if not draft_model:
        raise KeyError(
            "no draft_model configured; set draft_model knob to the path of a small "
            "GGUF (e.g. a 0.5-3B quant of the same family) to enable speculative decoding"
        )
    # A draft model needs its own VRAM (weights + KV). Don't add it to a full card.
    mem_ratio = m.get("mem_used_ratio")
    if mem_ratio is not None and mem_ratio >= 0.85:
        raise ValueError(
            "VRAM near full; loading a draft model would risk OOM. Free VRAM (smaller "
            "main quant / KV quant) before enabling speculative decoding."
        )
    draft_n = int(k.get("draft_n", k.get("draft", 16)))
    if draft_n < 1:
        # --draft 0 / negative is a nonsensical speculative-decode config; refuse
        # so the engine advises rather than launching a server with a bad flag.
        raise ValueError(f"draft token count must be >= 1, got {draft_n}")
    # Offload the draft model to the GPU too. Without -ngld the draft runs on the
    # CPU and serialises every draft step, which negates the whole speedup. A small
    # draft model has few layers, so "all on GPU" (999) is safe and correct.
    draft_ngl = int(k.get("draft_n_gpu_layers", 999))
    params: dict = {
        "model": k.get("model"),
        "restart_command": k.get("restart_command"),
        "prior_argv": k.get("prior_argv", []),
        "model_draft": str(draft_model),
        "n_gpu_layers_draft": draft_ngl,
        "draft": draft_n,
        # Flash attention (on) cuts KV bandwidth on the verifier pass so it costs
        # less per N-token batch; rendered with an explicit on/off value.
        "flash_attn": "on",
        "mlock": k.get("mlock", True),
        "drain_timeout_s": k.get("drain_timeout_s", 30.0),
    }
    # Optional, VRAM-free acceptance-tuning knobs — passed through only when the
    # operator set them (flag names match their llama.cpp build).
    for knob in ("draft_min", "draft_p_min"):
        if k.get(knob) is not None:
            params[knob] = k[knob]
    return params


def _build_fix_offload(ctx: ActionContext) -> dict:
    """Restart with the WHOLE model on the GPU. Refuses when it can't help.

    Raises (so the engine falls through to advise) when VRAM is already near full
    or the model provably won't fit — in those cases ``-ngl 999`` would OOM and
    the right answer is a smaller quant, which the monitor already advises."""
    m = ctx.metrics
    k = ctx.knobs
    mem_ratio = m.get("mem_used_ratio")
    if mem_ratio is not None and mem_ratio >= 0.90:
        raise ValueError("VRAM already near full; raising -ngl would risk OOM")
    model_gb = k.get("model_size_gb")
    vram_gb = k.get("vram_total_gb")
    if model_gb and vram_gb and float(model_gb) * 1.1 > float(vram_gb):
        raise ValueError("model does not fit VRAM; advise a smaller quant instead")
    return {
        "model": k.get("model"),
        "restart_command": k.get("restart_command"),
        "prior_argv": k.get("prior_argv", []),
        "n_gpu_layers": int(k.get("model_n_layers", 999)),
        "mlock": k.get("mlock", True),
        "drain_timeout_s": k.get("drain_timeout_s", 30.0),
    }


# -- specs -------------------------------------------------------------------

THERMAL_POWER_THROTTLE = ActionSpec(
    root_cause=RootCause.THERMAL_POWER_THROTTLE,
    kind=ActionKind.SET_POWER_LIMIT,
    action_class=ActionClass.NON_DISRUPTIVE,
    reversible=True,
    summary="Raise the GPU power limit so the SM clock can recover from throttle.",
    build_params=_build_power_limit,
    recovered=clock_recovered,
)

IDLE_ZOMBIE_PROCESS = ActionSpec(
    root_cause=RootCause.IDLE_ZOMBIE_PROCESS,
    kind=ActionKind.KILL_ORPHAN_PROCESS,
    action_class=ActionClass.NON_DISRUPTIVE,
    reversible=False,  # killing cannot be undone -> held to the irreversible guard
    summary="Kill the orphaned/zombie process holding the GPU (never the workload).",
    build_params=_build_kill_orphan,
    recovered=memory_freed,
    irreversible_guard=lambda ctx: (
        _orphan_pid(ctx) is not None and not ctx.protected.protects(_orphan_pid(ctx))
    ),
)

CPU_BOUND_PREPROCESSING = ActionSpec(
    root_cause=RootCause.CPU_BOUND_PREPROCESSING,
    kind=ActionKind.RENICE_PROCESS,
    action_class=ActionClass.NON_DISRUPTIVE,
    reversible=True,
    summary="Re-nice the data-loader worker procs to relieve CPU-bound preprocessing.",
    build_params=_build_renice,
    recovered=util_recovered,
)

MEMORY_FRAGMENTATION = ActionSpec(
    root_cause=RootCause.MEMORY_FRAGMENTATION,
    kind=ActionKind.FREE_STALE_CACHE,
    action_class=ActionClass.NON_DISRUPTIVE,
    reversible=False,
    summary="Free stale cached allocations held by a dead context to defragment VRAM.",
    build_params=_build_free_stale,
    recovered=memory_freed,
    irreversible_guard=lambda ctx: (
        _stale_pid(ctx) is not None and not ctx.protected.protects(_stale_pid(ctx))
    ),
)

DATA_PIPELINE_STARVATION = ActionSpec(
    root_cause=RootCause.DATA_PIPELINE_STARVATION,
    kind=ActionKind.DRAIN_AND_RESTART_WORKLOAD,
    action_class=ActionClass.DISRUPTIVE,
    reversible=True,
    summary="Restart the job with more DataLoader workers (needs checkpoint+drain).",
    build_params=_build_restart_dataloader,
    recovered=util_recovered,
    requires_checkpoint=True,
    requires_drain=True,
)

DISTRIBUTED_COMM_STALL = ActionSpec(
    root_cause=RootCause.DISTRIBUTED_COMM_STALL,
    kind=ActionKind.DRAIN_AND_RESTART_WORKLOAD,
    action_class=ActionClass.DISRUPTIVE,
    reversible=True,
    summary="Restart the job with tuned NCCL env to clear a distributed comm stall.",
    build_params=_build_restart_nccl,
    recovered=util_recovered,
    requires_checkpoint=True,
    requires_drain=True,
)

SUBOPTIMAL_RUNTIME_FLAGS = ActionSpec(
    root_cause=RootCause.SUBOPTIMAL_RUNTIME_FLAGS,
    kind=ActionKind.RESTART_LLAMA_SERVER,
    action_class=ActionClass.DISRUPTIVE,
    reversible=True,
    summary="Restart llama-server with tuned flags (-ngl/-b/--parallel/cache-type/--mlock).",
    build_params=_build_restart_llama,
    # Success is more tokens/sec, NOT more util%: a continuous-batching tune lifts
    # throughput while single-stream utilization barely moves (and a bad tune can
    # raise util while lowering tok/s). Verify on the metric that actually matters.
    recovered=throughput_recovered,
    requires_drain=True,
)

PARTIAL_GPU_OFFLOAD = ActionSpec(
    root_cause=RootCause.PARTIAL_GPU_OFFLOAD,
    kind=ActionKind.RESTART_LLAMA_SERVER,
    action_class=ActionClass.DISRUPTIVE,
    reversible=True,
    summary="Restart llama-server with the whole model on the GPU (-ngl 999).",
    build_params=_build_fix_offload,
    recovered=throughput_recovered,
    requires_drain=True,
)

SPEC_DECODE_AT_CEILING = ActionSpec(
    root_cause=RootCause.AT_PRACTICAL_CEILING,
    kind=ActionKind.RESTART_LLAMA_SERVER,
    action_class=ActionClass.DISRUPTIVE,
    reversible=True,
    summary=(
        "Enable speculative decoding (--model-draft / --draft) to push past the "
        "single-stream memory-bandwidth wall. Requires a draft_model knob; falls "
        "back to advise-only when none is configured."
    ),
    build_params=_build_spec_decode,
    recovered=throughput_recovered,
    requires_drain=True,
)


ALL_STRATEGIES: list[ActionSpec] = [
    THERMAL_POWER_THROTTLE,
    IDLE_ZOMBIE_PROCESS,
    CPU_BOUND_PREPROCESSING,
    MEMORY_FRAGMENTATION,
    DATA_PIPELINE_STARVATION,
    DISTRIBUTED_COMM_STALL,
    SUBOPTIMAL_RUNTIME_FLAGS,
    PARTIAL_GPU_OFFLOAD,
    SPEC_DECODE_AT_CEILING,
]


def default_registry() -> ActionRegistry:
    """The registry wired with all built-in strategies."""
    return ActionRegistry().register_all(ALL_STRATEGIES)
