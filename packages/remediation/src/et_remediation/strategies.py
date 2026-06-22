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


def _build_restart_llama(ctx: ActionContext) -> dict:
    return {
        "model": ctx.knobs.get("model"),
        "restart_command": ctx.knobs.get("restart_command"),
        "prior_argv": ctx.knobs.get("prior_argv", []),
        "n_gpu_layers": ctx.knobs.get("n_gpu_layers", 999),
        "parallel": ctx.knobs.get("parallel", 4),
        "cache_type_k": ctx.knobs.get("cache_type_k", "q8_0"),
        "cache_type_v": ctx.knobs.get("cache_type_v", "q8_0"),
        "mlock": ctx.knobs.get("mlock", True),
        "drain_timeout_s": ctx.knobs.get("drain_timeout_s", 30.0),
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
    summary="Restart llama-server with tuned flags (-ngl/-t/-b/cache-type/--mlock).",
    build_params=_build_restart_llama,
    recovered=util_recovered,
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
]


def default_registry() -> ActionRegistry:
    """The registry wired with all built-in strategies."""
    return ActionRegistry().register_all(ALL_STRATEGIES)
