"""Expanded ground-truth harness: verdict BOUNDARIES + MIXED workloads.

RESULTS (T4 Colab run)
======================
TODO(run-on-t4): paste the table produced by
``python -m accuracy.ground_truth_expanded`` once it has executed on a Colab
T4 (or equivalent CUDA host). Until that table is here, the engine's accuracy
on these boundary / contention workloads is UNMEASURED. The numeric targets in
each builder's docstring are *intended* shares, reasoned from the engine's
thresholds — they are NOT measured values, and a real T4 will land somewhere
near (not exactly on) them. That is the whole reason the harness prints the
ACHIEVED shares: so a mismatch can be attributed to the right cause.

Why this module exists (vs ``accuracy.ground_truth``)
=====================================================
``ground_truth.py`` plants ONE clean single-cause workload per verdict and
checks the engine names it. That validates each detector in isolation. It does
NOT exercise the part of the engine the ``fix/verdict-precedence-bugs`` branch
actually changed: the **dominant-cause competition** and the strengthened
**HEALTHY gate**. Those only matter when

  * a cause sits right on the edge of its firing threshold (does the detector
    fire across a RANGE, or only at one hand-tuned point?), or
  * TWO causes are present at once and the engine must pick the dominant one
    (does the competition pick the larger idle share — and do specific causes
    correctly outrank the generic DataLoader fallback, and does the
    kernel-launch SYMPTOM correctly lose to any real idle cause?).

So this harness adds three kinds of coverage, all as REAL cuda workloads:

1. SEVERITY SWEEP — for dataloader / sync / checkpoint / pcie, three variants
   each (mild / moderate / severe) tuned by stall magnitude. moderate + severe
   must produce the SAME bottleneck verdict (the detector fires across a range,
   not at one point); the deliberately-mild variant is tuned BELOW threshold
   and its expected verdict is HEALTHY or UNKNOWN — documented per builder with
   the reason it falls below.

2. MIXED / COMPETING — the key new coverage. Two causes present at once, with a
   known intended winner BY CONSTRUCTION:
     * dataloader-dominant + tiny kernels   -> DATALOADER_BOUND (the Bug-1 case:
       the tiny-kernel SYMPTOM must lose to the dataloader CAUSE).
     * sync-dominant + moderate pcie copies -> SYNC_BOUND   (sync share > pcie).
     * pcie-dominant + some sync calls       -> PCIE_BOUND   (pcie share > sync).
     * checkpoint-dominant + dataloader      -> CHECKPOINT_BOUND (a SPECIFIC
       cause outranks the generic DataLoader fallback).
   Each builder documents the intended share ordering.

3. NEAR-BOUNDARY — one workload tuned just ABOVE a threshold (must fire) and one
   just BELOW (must NOT fire that verdict). Covered for sync (the util < 0.70
   boundary that caused Bug-2) and dataloader (the dl_share >= 0.20 boundary).

Honesty contract (identical to ``ground_truth.py``)
===================================================
* Every workload triggers its pattern through real workload mechanics — no
  post-hoc trace edits, no synthetic event injection.
* Do NOT tune the engine to pass these. A mismatch is a finding.
* Real GPU workloads are noisy, so a mismatch has TWO possible causes and the
  report distinguishes them:
    - ENGINE bug: the workload DID achieve the intended firing condition
      (the expected cause crossed its own threshold — reported as
      ``cause_fired=yes``) but the engine still picked a different verdict.
    - CONSTRUCTION issue: the workload did NOT achieve the intended shares
      (``cause_fired=no``) — e.g. a T4's PCIe bandwidth pushed pcie_ratio under
      0.50, so the trace genuinely is not pcie-bound. That is a workload bug to
      fix in THIS file, not an engine accuracy gap.
  Both the ACHIEVED share ordering and ``cause_fired`` are printed for every
  mismatch so the reader can tell which it is without re-running.

How to run
==========
Colab cell::

    !pip install -q torch
    !pip install -q -e /content/ET/packages/engine
    !cd /content/ET/packages/engine && python -m accuracy.ground_truth_expanded

Local (CUDA box, editable engine install)::

    cd packages/engine
    uv run python -m accuracy.ground_truth_expanded

Importable too::

    from accuracy.ground_truth_expanded import run_all_expanded
    report = run_all_expanded()

Design notes
============
* Torch is imported LAZILY inside each builder; the module imports cleanly on a
  torch-less host so ``tests/test_ground_truth_expanded_smoke.py`` can verify
  structure without a GPU. Same pattern as ``accuracy.ground_truth`` and
  ``gpu_doctor_agent.torch_source``.
* Each builder runs ``diagnose_with_stats`` and the report prints the decisive
  shares, so a mismatch is diagnosable from the printed table alone.
* NCCL is NOT exercised here (needs >=2 GPUs; see ``ground_truth.py``).
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable

from gpu_doctor_engine import Verdict, diagnose_with_stats, load_trace

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy torch availability flag — set at import time, NEVER raises.
# (Mirrors accuracy.ground_truth verbatim so both harnesses degrade identically.)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - trivial import branch
    import torch  # type: ignore[import-not-found]  # noqa: F401

    _TORCH_AVAILABLE: bool = True
except Exception:  # ImportError, or torch present but broken
    _TORCH_AVAILABLE = False


class TorchUnavailable(RuntimeError):
    """Torch or CUDA was missing when a builder was invoked.

    Same shape as ``accuracy.ground_truth.TorchUnavailable`` and
    ``gpu_doctor_agent.torch_source``: one well-named error to catch so CI /
    no-GPU dev runs degrade gracefully instead of leaking a bare ImportError.
    """


def _require_cuda() -> None:
    """Raise ``TorchUnavailable`` if torch isn't importable or no GPU is visible."""
    if not _TORCH_AVAILABLE:
        raise TorchUnavailable("torch is not installed on this host")
    import torch  # type: ignore[import-not-found]

    if not torch.cuda.is_available():
        raise TorchUnavailable("no CUDA device visible to torch")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class ExpandedResult:
    """One variant's outcome.

    ``intended_order`` / ``achieved_order`` are the documented and the actually
    measured share orderings — populated for every variant, but only really
    meaningful for the mixed / boundary cases where the *ordering* is the point.
    ``cause_fired`` answers "did the workload achieve the intended firing
    condition for ``expected``?" (None for HEALTHY / UNKNOWN, where there is no
    cause to fire) — the engine-bug-vs-construction-issue discriminator.
    """

    name: str
    group: str
    expected: Verdict
    actual: Verdict
    confidence: float
    match: bool
    decisive_metric: str
    intended_order: str
    achieved_order: str
    cause_fired: bool | None
    stats: dict
    trace_path: str | None = None
    error: str | None = None


@dataclass
class ExpandedReport:
    results: list[ExpandedResult]
    correct: int
    total: int
    mismatches: list[ExpandedResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Share helpers — turn a stats dict into the engine-relevant shares.
# ---------------------------------------------------------------------------
def _shares(stats: dict) -> dict[str, float]:
    """The five competing shares the dominant-cause competition ranks on.

    Note the denominators differ ON PURPOSE — they mirror exactly what the
    engine compares: ``pcie_ratio`` is memcpy / GPU-active, the other four are
    cause-time / idle-time. The competition sorts these raw floats, so the
    report shows them on the same scale the engine uses.
    """
    idle = max(stats.get("idle_us", 1), 1)
    active = max(stats.get("gpu_active_us", 1), 1)
    return {
        "dl_share": stats.get("dataloader_us", 0) / idle,
        "sync_fraction": float(stats.get("sync_fraction", 0.0)),
        "ckpt_share": stats.get("checkpoint_us", 0) / idle,
        "nccl_share": stats.get("nccl_us", 0) / idle,
        "pcie_ratio": stats.get("gpu_memcpy_time_us", 0) / active,
    }


def _share_ordering(stats: dict) -> str:
    """Descending ranking of the five shares, e.g. 'sync_fraction=0.55 > ...'."""
    ranked = sorted(_shares(stats).items(), key=lambda kv: kv[1], reverse=True)
    return " > ".join(f"{k}={v:.2f}" for k, v in ranked)


def _expected_cause_fired(expected: Verdict, stats: dict) -> bool | None:
    """Did the workload make ``expected``'s OWN firing condition true?

    Uses the engine's exact firing predicate for each cause (see
    ``diagnose._diagnose_core``). Returns None for HEALTHY / UNKNOWN where there
    is no single cause to fire. This is the engine-bug-vs-construction-issue
    discriminator: a mismatch with ``cause_fired=True`` is an engine bug; with
    ``cause_fired=False`` the workload simply did not achieve its intended
    shares (fix the workload, not the engine).
    """
    idle = max(stats.get("idle_us", 1), 1)
    if expected == Verdict.DATALOADER_BOUND:
        return stats.get("dataloader_us", 0) > 0 and (
            stats.get("dataloader_us", 0) / idle >= 0.20
        )
    if expected == Verdict.SYNC_BOUND:
        return (
            stats.get("util", 1.0) < 0.70
            and float(stats.get("sync_fraction", 0.0)) >= 0.25
            and stats.get("avg_kernel_dur", 0.0) >= 50
        )
    if expected == Verdict.PCIE_BOUND:
        active = max(stats.get("gpu_active_us", 1), 1)
        pcie_ratio = stats.get("gpu_memcpy_time_us", 0) / active
        s30 = stats.get("memcpy_us", 0) / idle
        return pcie_ratio >= 0.50 or s30 >= 0.30
    if expected == Verdict.CHECKPOINT_BOUND:
        return stats.get("checkpoint_dtoh_count", 0) >= 50 or (
            stats.get("checkpoint_us", 0) / idle >= 0.25
        )
    if expected == Verdict.NCCL_BOUND:
        return stats.get("nccl_us", 0) / idle >= 0.30
    return None  # HEALTHY / UNKNOWN: no single cause to fire


# ---------------------------------------------------------------------------
# Profile + diagnose helper
# ---------------------------------------------------------------------------
def _run_variant(
    name: str,
    group: str,
    expected: Verdict,
    run_loop: Callable[[], None],
    decisive_fn: Callable[[dict], str],
    intended_order: str = "",
) -> ExpandedResult:
    """Profile ``run_loop``, export the trace, diagnose it, build an ExpandedResult.

    The trace file is left on disk so a mismatch can be replayed with the CLI.
    Same profiling shape as ``accuracy.ground_truth._profile_and_diagnose``.
    """
    _require_cuda()
    import torch  # type: ignore[import-not-found]
    from torch.profiler import ProfilerActivity, profile  # type: ignore

    fd, path = tempfile.mkstemp(prefix=f"gt_expanded_{name}_", suffix=".json")
    os.close(fd)
    trace_path = Path(path)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        run_loop()
    # Drain in-flight CUDA work BEFORE exporting so every launched kernel
    # appears in the trace with a real duration.
    torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))

    trace = load_trace(trace_path)
    diag, stats = diagnose_with_stats(trace)
    achieved = _share_ordering(stats)
    # Print the achieved shares as the workload finishes so a Colab user sees
    # whether the intended share ordering was actually produced — independent
    # of the verdict. (The honesty contract: noisy GPU workloads may not hit
    # their target shares; this line is how you tell.)
    print(f"  [{name}] achieved: {achieved}  ({decisive_fn(stats)})")

    return ExpandedResult(
        name=name,
        group=group,
        expected=expected,
        actual=diag.verdict,
        confidence=diag.confidence,
        match=(diag.verdict == expected),
        decisive_metric=decisive_fn(stats),
        intended_order=intended_order,
        achieved_order=achieved,
        cause_fired=_expected_cause_fired(expected, stats),
        stats=stats,
        trace_path=str(trace_path),
    )


# ---------------------------------------------------------------------------
# Shared builder for CPU<->GPU-sync workloads with a CONTROLLED utilisation.
# ---------------------------------------------------------------------------
def _synced_matmul_run_loop(
    matmul_dim: int,
    target_util: float,
    n_steps: int = 120,
    transfer_mb: float = 0.0,
) -> Callable[[], None]:
    """Build a run_loop for a sync-bound workload that pins util to ``target_util``.

    SYNC_BOUND needs THREE things at once and on a real GPU they pull against each
    other: avg_kernel_dur >= 50us (non-tiny kernels), util < 0.70 (the GPU is
    starved), and sync_fraction high. This builder makes all three hold by
    construction, decoupling them:

    * Each step runs ONE ``matmul_dim`` x ``matmul_dim`` matmul -- the ONLY compute
      kernel, so ``avg_kernel_dur`` equals its duration (>= 50us for dim >= 768).
      The FIRST T4 run of this harness used ``c.sum().item()``; the ``.sum()``
      reduction kernel dragged the AVERAGE kernel duration to ~35us (< 50) and the
      sync rule correctly refused to fire (UNKNOWN). We drop the reduction entirely.
    * The stream sync is ``torch.cuda.synchronize()`` (matches SYNC_PATTERNS). It
      adds NO kernel and NO device->host copy, so it cannot trip the kernel-launch
      detector or the checkpoint DtoH detector -- the latter matters for the
      sub-0.70 dead-zone case, where sync does NOT fire and a stray DtoH burst from
      ``.item()`` could otherwise win as a phantom CHECKPOINT_BOUND.
    * util is pinned to ``target_util`` independent of GPU speed: we MEASURE one
      matmul up front and ``time.sleep`` ``gemm_s * (1/target_util - 1)`` after the
      sync each step. The sleep models host-side work after a blocking sync (the
      realistic cause of GPU starvation in SYNC_BOUND); because the sync ends
      < 500us before that idle starts, the engine attributes the whole sleep to
      sync, so ``sync_fraction`` stays ~1.0. The ACHIEVED util lands a hair below
      target (the launch gap adds a little extra idle). A LARGER matmul makes the
      sized sleep land in ``time.sleep``'s reliable (> ~1ms) range -- which is why
      the tight dead-zone case must use a large matmul, while the util < 0.70 cases
      are robust even if a sub-ms sleep rounds up (more idle only lowers util).
    * ``transfer_mb`` > 0 adds a host->device copy per step (a secondary PCIe
      signal) for the mixed sync-vs-pcie contention case.
    """
    import time

    import torch  # type: ignore[import-not-found]

    device = torch.device("cuda")
    a = torch.randn(matmul_dim, matmul_dim, device=device)
    b = torch.randn(matmul_dim, matmul_dim, device=device)
    cpu_buf = None
    if transfer_mb > 0:
        side = max(int((transfer_mb * 1e6 / 4) ** 0.5), 1)  # float32 = 4 bytes
        cpu_buf = torch.randn(side, side)

    # Measure one matmul (after warm-up) to size the post-sync sleep for target_util.
    # This runs OUTSIDE the profiler (the builder calls us before _run_variant), so
    # the warm-up matmuls never enter the diagnosed trace.
    for _ in range(3):
        _ = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        _ = a @ b
    torch.cuda.synchronize()
    gemm_s = max((time.perf_counter() - t0) / 10, 1e-5)
    sleep_s = max(gemm_s * (1.0 / target_util - 1.0), 0.0)

    def run_loop() -> None:
        for _ in range(n_steps):
            if cpu_buf is not None:
                _ = cpu_buf.cuda(non_blocking=False)  # secondary PCIe signal
            _ = a @ b  # the ONLY compute kernel -> avg_kernel_dur == its duration
            torch.cuda.synchronize()  # blocking sync; adds no kernel, no DtoH copy
            if sleep_s > 0:
                # Host-side post-sync work. The idle it creates is sync-attributed
                # (the sync ended < 500us before this idle starts).
                time.sleep(sleep_s)
        torch.cuda.synchronize()

    return run_loop


# ===========================================================================
# 1. SEVERITY SWEEP
# ===========================================================================
# For each cause we vary ONE knob (stall magnitude) across three levels. The
# moderate + severe variants must produce the SAME bottleneck verdict — that is
# the point: the detector must fire across a RANGE, not at one tuned point. The
# mild variant is deliberately pushed BELOW threshold; its expected verdict
# (HEALTHY or UNKNOWN) and the reason are documented in each builder.

_SEVERITY_LEVELS = ("mild", "moderate", "severe")


def plant_dataloader_severity(level: str) -> ExpandedResult:
    """DataLoader stall swept by sleep magnitude — severity is a UTIL sweep.

    Construction
    ------------
    ``num_workers=0`` ``Dataset`` that sleeps ``S`` ms per sample, plus a fixed
    block of real GPU compute (four 2048x2048 matmuls) per step. Because the
    DataLoader fetch is the ONLY source of GPU idle, ``dl_share`` is ~1.0 at
    EVERY level — what the sleep magnitude actually moves is UTIL
    (= GPU-busy / wall ≈ compute_time / (compute_time + sleep)). So this sweep
    really tests the HEALTHY-gate util tiers, which is exactly the Bug-2 region.

    Intended levels
    ---------------
    * mild     S=1ms/sample with a HEAVY compute block (four 4096x4096 matmuls,
      ~tens of ms) that dwarfs the fetch stall -> util >= 0.70. Expected HEALTHY.
      WHY: a ~1ms fetch behind multi-ms compute is healthy OVERLAPPED prefetch,
      not a bottleneck -- the healthy_70_no_dominant gate is deliberately UNGATED
      on dl_share (and healthy_85 if util clears 0.85). The FIRST T4 run used a
      2ms stall with a LIGHT batch matmul (~1ms compute) and util collapsed to
      0.10 -- genuinely dataloader-bound, NOT mild. Fixed by shrinking the stall
      and making compute dominant.
    * moderate S=15ms/sample with a LIGHT compute block -> util ~0.1-0.3 (<0.45).
      Expected DATALOADER_BOUND.
    * severe   S=50ms/sample with a LIGHT compute block -> util ~0.05. DATALOADER_BOUND.

    dl_share is ~1.0 at every level (the fetch stall is the only idle), so even if
    moderate's util drifts up toward 0.70 it cannot be called HEALTHY at the
    borderline tier (borderline_healthy_45 requires dl_share < 0.40).
    """
    _require_cuda()
    import time

    import torch  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader, Dataset  # type: ignore

    if level == "mild":
        sleep_s, expected = 0.001, Verdict.HEALTHY
        compute_dim, n_matmuls = 4096, 4  # heavy: dwarfs the ~1ms fetch stall
    else:
        sleep_s = 0.015 if level == "moderate" else 0.050
        expected = Verdict.DATALOADER_BOUND
        compute_dim, n_matmuls = 2048, 4  # light: the big stall dominates wall

    class SleepDataset(Dataset):
        def __len__(self) -> int:
            return 24

        def __getitem__(self, idx: int) -> "torch.Tensor":
            time.sleep(sleep_s)  # << the swept DataLoader stall
            return torch.randn(64, 64)

    device = torch.device("cuda")
    loader = DataLoader(SleepDataset(), batch_size=4, num_workers=0)
    # Device-resident square matmul sets util via compute magnitude, independent
    # of the (tiny) batch -- mild's 4096x4096 block is what keeps util high.
    cmat = torch.randn(compute_dim, compute_dim, device=device)

    def run_loop() -> None:
        for batch in loader:
            x = batch.to(device, non_blocking=False)
            h = cmat
            for _ in range(n_matmuls):
                h = torch.relu(h @ cmat)
            _ = h.sum() + x.sum()
        torch.cuda.synchronize()

    def _decisive(s: dict) -> str:
        dl = s.get("dataloader_us", 0)
        idle = max(s.get("idle_us", 1), 1)
        return f"dl_share={dl / idle:.2f} util={s.get('util', 0):.2f}"

    return _run_variant(
        name=f"dataloader_{level}",
        group="severity:dataloader",
        expected=expected,
        run_loop=run_loop,
        decisive_fn=_decisive,
    )


def plant_sync_severity(level: str) -> ExpandedResult:
    """CPU<->GPU sync stalls swept by how starved the GPU is (util).

    Construction
    ------------
    SYNC_BOUND requires non-tiny kernels (avg_kernel_dur >= 50us) AND a starved
    GPU (util < 0.70) AND high sync_fraction. The FIRST T4 run of this harness
    mis-constructed moderate/severe: a 512x512 matmul with ``c.sum().item()`` put
    the average kernel duration at ~35us (the ``.sum()`` reduction kernel dragged
    it under 50), so the sync rule correctly refused to fire and the verdict was
    UNKNOWN -- a workload bug, not an engine bug. The fix routes moderate/severe
    through ``_synced_matmul_run_loop`` (see its docstring): one wide 2048x2048
    matmul as the only kernel (avg_kernel_dur ~ms, comfortably > 50us), a
    ``torch.cuda.synchronize()`` sync (no reduction kernel), and a measured
    post-sync sleep that pins util below 0.70 while keeping sync_fraction ~1.0.

    Intended levels
    ---------------
    * mild     1024x1024 matmul x2/step, ``.item()`` only every 16th step
      -> GPU runs back-to-back, util >= 0.85. Expected HEALTHY. WHY: an occasional
      sync does not starve the GPU; healthy_85 fires (it ignores sync_fraction).
      (Unchanged -- mild was correct on T4.)
    * moderate util pinned to ~0.50 (< 0.70). Expected SYNC_BOUND.
    * severe   util pinned to ~0.25 (deeper starvation). Expected SYNC_BOUND.

    moderate and severe must produce the SAME verdict (SYNC_BOUND) -- the detector
    firing across a util RANGE is the point.
    """
    _require_cuda()
    import torch  # type: ignore[import-not-found]

    if level == "mild":
        device = torch.device("cuda")
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)

        def run_loop() -> None:
            acc = a
            for step in range(320):
                acc = acc @ b
                acc = torch.relu(acc)
                if step % 16 == 0:  # rare sync — does not starve the GPU
                    _ = acc.sum().item()
            torch.cuda.synchronize()

        expected = Verdict.HEALTHY
    else:
        # 2048x2048 matmul -> avg_kernel_dur ~ms (>> 50us guard); util pinned by
        # the measured post-sync sleep. moderate ~0.50, severe ~0.25 (both < 0.70).
        target_util = 0.50 if level == "moderate" else 0.25
        run_loop = _synced_matmul_run_loop(matmul_dim=2048, target_util=target_util)
        expected = Verdict.SYNC_BOUND

    def _decisive(s: dict) -> str:
        return (
            f"sync_fraction={s.get('sync_fraction', 0):.2f} "
            f"avg={s.get('avg_kernel_dur', 0):.0f}us util={s.get('util', 0):.2f}"
        )

    return _run_variant(
        name=f"sync_{level}",
        group="severity:sync",
        expected=expected,
        run_loop=run_loop,
        decisive_fn=_decisive,
    )


def plant_checkpoint_severity(level: str) -> ExpandedResult:
    """torch.save checkpointing swept by saved-tensor count (DtoH-Pageable events).

    Construction
    ------------
    The primary checkpoint signal is the count of ``Memcpy DtoH (Device ->
    Pageable)`` events ``torch.save`` emits (one per leaf tensor). The strong
    rule fires at >= 50. Severity = how many tensors x how many saves.

    Intended levels
    ---------------
    * mild     1-layer model (2 tensors), 1 save, preceded by a compute-heavy
      "training" loop (real matmuls). -> dtoh ~2 (< 50) AND ckpt_share < 0.25,
      util high. Expected HEALTHY. WHY: a 2-tensor checkpoint is below the
      strong signal and below the fallback share; compute dominates -> HEALTHY.
      A cheap save is not a bottleneck.
    * moderate 30-layer model (60 tensors), 1 save -> dtoh ~60 (>= 50).
      Expected CHECKPOINT_BOUND.
    * severe   30-layer model, 3 saves -> dtoh ~180, larger ckpt_share, higher
      confidence. Expected CHECKPOINT_BOUND.

    moderate/severe keep util < 0.45 (the synchronous save dominates wall clock),
    so the HEALTHY gate cannot claim them before the checkpoint rule runs.
    Some torch builds copy state_dict via PINNED (not pageable) memory; then the
    DtoH-Pageable count MISSES and the name-pattern fallback (ckpt_share >= 0.25)
    must carry it — the achieved dtoh_count in the report tells you which path
    fired (documented honestly in ground_truth.plant_checkpoint_bound too).
    """
    _require_cuda()
    import torch  # type: ignore[import-not-found]

    device = torch.device("cuda")

    if level == "mild":
        # Compute-heavy "training" so util stays high; the save is trivial.
        big = torch.randn(2048, 2048, device=device)
        model = torch.nn.Linear(64, 64).to(device)  # 2 tensors -> ~2 dtoh
        n_saves = 1
        expected = Verdict.HEALTHY

        def warmup() -> None:
            acc = big
            for _ in range(40):
                acc = acc @ big
            _ = acc.sum()
    else:
        layers: list[torch.nn.Module] = []
        for _ in range(30):  # 30 Linear -> 60 parameter tensors
            layers.append(torch.nn.Linear(64, 64))
            layers.append(torch.nn.ReLU())
        model = torch.nn.Sequential(*layers).to(device)
        n_saves = 3 if level == "severe" else 1
        expected = Verdict.CHECKPOINT_BOUND

        def warmup() -> None:
            x = torch.randn(32, 64, device=device)
            opt = torch.optim.SGD(model.parameters(), lr=1e-3)
            for _ in range(20):  # >= 5 kernels so the warmup guard is bypassed
                opt.zero_grad(set_to_none=True)
                out = model(x)
                loss = out.sum()
                loss.backward()
                opt.step()

    def run_loop() -> None:
        warmup()
        torch.cuda.synchronize()
        fd, ckpt_path = tempfile.mkstemp(suffix=".pt", prefix="gt_expanded_ckpt_")
        os.close(fd)
        try:
            for _ in range(n_saves):
                torch.save(model.state_dict(), ckpt_path)  # << the planted stall
        finally:
            try:
                os.unlink(ckpt_path)
            except OSError:
                pass

    def _decisive(s: dict) -> str:
        idle = max(s.get("idle_us", 1), 1)
        return (
            f"dtoh={s.get('checkpoint_dtoh_count', 0)} "
            f"ckpt_share={s.get('checkpoint_us', 0) / idle:.2f} "
            f"util={s.get('util', 0):.2f}"
        )

    return _run_variant(
        name=f"checkpoint_{level}",
        group="severity:checkpoint",
        expected=expected,
        run_loop=run_loop,
        decisive_fn=_decisive,
    )


def plant_pcie_severity(level: str) -> ExpandedResult:
    """Host<->device transfers swept by transfer-size / compute ratio.

    Construction
    ------------
    Per iteration: copy a CPU tensor to the GPU, do some compute, copy back.
    ``pcie_ratio = gpu_memcpy_time / (gpu_kernel_time + gpu_memcpy_time)``; the
    rule fires at >= 0.50. Severity = transfer bytes vs compute work.

    Intended levels
    ---------------
    * mild     4MB transfer + a 4096x4096 matmul (heavy compute) -> pcie_ratio
      ~0.15 (< 0.50), util high. Expected HEALTHY. WHY: a little H2D copying
      against compute-dominated steps is normal; pcie does not fire and the GPU
      stays busy -> HEALTHY.
    * moderate 64MB transfer + a 1024x1024 matmul -> pcie_ratio ~0.55.
      Expected PCIE_BOUND.
    * severe   64MB transfer + only a relu (tiny compute) -> pcie_ratio ~0.85.
      Expected PCIE_BOUND (high confidence).

    PCIe bandwidth on a shared Colab T4 varies 6-12 GB/s, so the moderate ratio
    is the least robust point — if it slips under 0.50 the verdict flips and the
    achieved pcie_ratio column shows it was a construction (bandwidth) outcome,
    not an engine miss.
    """
    _require_cuda()
    import torch  # type: ignore[import-not-found]

    device = torch.device("cuda")

    if level == "mild":
        cpu_buf = torch.randn(1024, 1024)  # 4 MB
        w = torch.randn(4096, 4096, device=device)

        def run_loop() -> None:
            for _ in range(20):
                _ = cpu_buf.cuda(non_blocking=False)
                _ = w @ w  # heavy compute dominates GPU-active time
            torch.cuda.synchronize()

        expected = Verdict.HEALTHY
    elif level == "moderate":
        cpu_buf = torch.randn(4096, 4096)  # 64 MB
        w = torch.randn(1024, 1024, device=device)

        def run_loop() -> None:
            for _ in range(30):
                gpu = cpu_buf.cuda(non_blocking=False)
                _ = w @ w  # moderate compute -> ratio near the 0.50 edge
                _ = gpu.cpu()
            torch.cuda.synchronize()

        expected = Verdict.PCIE_BOUND
    else:  # severe
        cpu_buf = torch.randn(4096, 4096)  # 64 MB

        def run_loop() -> None:
            for _ in range(30):
                gpu = cpu_buf.cuda(non_blocking=False)
                torch.relu_(gpu)  # tiny compute -> transfers dominate
                _ = gpu.cpu()
            torch.cuda.synchronize()

        expected = Verdict.PCIE_BOUND

    def _decisive(s: dict) -> str:
        memcpy = s.get("gpu_memcpy_time_us", 0)
        active = max(s.get("gpu_active_us", 1), 1)
        return f"pcie_ratio={memcpy / active:.2f} util={s.get('util', 0):.2f}"

    return _run_variant(
        name=f"pcie_{level}",
        group="severity:pcie",
        expected=expected,
        run_loop=run_loop,
        decisive_fn=_decisive,
    )


# ===========================================================================
# 2. MIXED / COMPETING WORKLOADS  (the key new coverage)
# ===========================================================================
# Two causes present at once; the intended winner is the cause with the larger
# idle share BY CONSTRUCTION. Each builder documents the intended share ordering
# in its docstring; the harness prints intended vs achieved so a wrong verdict
# can be split into "engine picked the wrong winner" vs "the workload didn't
# actually produce the intended ordering".


def plant_mixed_dataloader_vs_tiny_kernels() -> ExpandedResult:
    """dataloader-dominant + tiny kernels -> DATALOADER_BOUND (the Bug-1 case).

    Intended share ordering
    ------------------------
        dl_share (~0.85+, the DataLoader sleep is essentially all the idle)
        >> everything else; tiny_kernel_ratio ~1.0 but kernel-launch is a
        SYMPTOM, not an idle cause.

    Why DATALOADER_BOUND
    --------------------
    A ``num_workers=0`` dataset sleeps 30ms/sample (the dominant idle) while each
    step also fires ~300 sub-microsecond ``add_`` kernels (tiny_kernel_ratio ~1.0,
    util low). Under first-match-wins this was mislabelled KERNEL_LAUNCH_BOUND
    (Bug 1). The dominant-cause competition must now name the CAUSE (DataLoader,
    an idle cause that fires dl_fired) over the SYMPTOM (kernel_launch_tiny,
    which can only win when NO idle cause fired). This is the real-workload twin
    of test_dataloader_dominant_beats_tiny_kernels.
    """
    _require_cuda()
    import time

    import torch  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader, Dataset  # type: ignore

    class SlowTinyDataset(Dataset):
        def __len__(self) -> int:
            return 16

        def __getitem__(self, idx: int) -> "torch.Tensor":
            time.sleep(0.030)  # dominant idle: dataloader stall
            return torch.randn(8, 8)

    device = torch.device("cuda")
    loader = DataLoader(SlowTinyDataset(), batch_size=4, num_workers=0)
    tiny = torch.ones(1, device=device)

    def run_loop() -> None:
        for _ in loader:
            for _ in range(300):  # the tiny-kernel SYMPTOM
                tiny.add_(1.0)
        torch.cuda.synchronize()

    def _decisive(s: dict) -> str:
        idle = max(s.get("idle_us", 1), 1)
        return (
            f"dl_share={s.get('dataloader_us', 0) / idle:.2f} "
            f"tiny_ratio={s.get('tiny_kernel_ratio', 0):.2f} "
            f"util={s.get('util', 0):.2f}"
        )

    return _run_variant(
        name="mixed_dataloader_vs_tiny_kernels",
        group="mixed",
        expected=Verdict.DATALOADER_BOUND,
        run_loop=run_loop,
        decisive_fn=_decisive,
        intended_order="dl_share >> (kernel_launch is a symptom, not an idle cause)",
    )


def plant_mixed_sync_vs_pcie() -> ExpandedResult:
    """sync-dominant + moderate pcie copies -> SYNC_BOUND (sync share > pcie share).

    Intended share ordering
    ------------------------
        sync_fraction (~1.0) > pcie_ratio (~0.55). Both causes FIRE (pcie_ratio
        clears 0.50; sync clears its guards), and sync wins the dominant-cause
        competition on the larger share -- genuine contention that sync wins.

    Why SYNC_BOUND
    --------------
    The FIRST T4 run of this harness got this BACKWARDS: an 8MB copy + a 512x512
    matmul + ``c.sum().item()`` achieved pcie_ratio=0.83 > sync_fraction=0.75, so
    PCIE_BOUND was the CORRECT verdict for that workload. This is the rebuild
    (option a -- make sync genuinely dominate, not flip the expectation):

    * The sync stall is built via ``_synced_matmul_run_loop``: a measured post-sync
      sleep makes the post-sync idle dominate, so sync_fraction ~1.0 -- larger than
      ANY pcie_ratio (which is bounded by 1.0). Sync therefore wins the share
      competition BY CONSTRUCTION.
    * The H2D copy is sized (~4MB) so its transfer time is comparable to the
      1024x1024 matmul, putting pcie_ratio near ~0.55 -- above 0.50, so pcie_ratio_50
      genuinely FIRES and the competition is real (not a walkover). gpu_active is
      large (matmul + memcpy over 120 steps) so the pcie active-time floor is moot.
    * util is pinned well below 0.70 by the sleep, so the sync rule fires.

    If a fast link pushes pcie_ratio lower (pcie may not even fire), sync still
    wins; if a slow link pushes it higher, sync still wins because sync_fraction
    ~1.0 > pcie_ratio. The report prints both so the achieved ordering is checkable.
    """
    _require_cuda()

    # 1024x1024 matmul (avg_kernel_dur >> 50us) + a ~4MB H2D copy/step (pcie_ratio
    # ~0.55, fires). target_util 0.40 on the matmul -> achieved util < 0.70 even
    # with the copy added to GPU-active time, so the sync rule fires.
    run_loop = _synced_matmul_run_loop(
        matmul_dim=1024, target_util=0.40, transfer_mb=4.0
    )

    def _decisive(s: dict) -> str:
        active = max(s.get("gpu_active_us", 1), 1)
        return (
            f"sync_fraction={s.get('sync_fraction', 0):.2f} "
            f"pcie_ratio={s.get('gpu_memcpy_time_us', 0) / active:.2f} "
            f"util={s.get('util', 0):.2f}"
        )

    return _run_variant(
        name="mixed_sync_vs_pcie",
        group="mixed",
        expected=Verdict.SYNC_BOUND,
        run_loop=run_loop,
        decisive_fn=_decisive,
        intended_order="sync_fraction (~1.0) > pcie_ratio (~0.55)",
    )


def plant_mixed_pcie_vs_sync() -> ExpandedResult:
    """pcie-dominant + some sync calls -> PCIE_BOUND (pcie share > sync share).

    Intended share ordering
    ------------------------
        pcie_ratio (~0.70) > sync_fraction (~0.30). pcie_ratio_50 fires and wins
        the competition on the larger share.

    Why PCIE_BOUND
    --------------
    Each step: a large 64MB H2D + DtoH copy pair (the dominant pcie signal) with
    only a tiny 256x256 matmul, and ``.item()`` every 4th step (a secondary sync
    signal). Transfers dominate GPU-active time so pcie_ratio clears 0.50 with a
    large margin. The tiny matmul also means avg_kernel_dur is likely < 50us, so
    the sync rule's guard may keep sync from firing at all — either way pcie's
    share is larger and pcie wins. The report's achieved pcie_ratio vs
    sync_fraction makes the winner auditable.
    """
    _require_cuda()
    import torch  # type: ignore[import-not-found]

    device = torch.device("cuda")
    cpu_buf = torch.randn(4096, 4096)  # 64 MB -> dominant pcie
    a = torch.randn(256, 256, device=device)
    b = torch.randn(256, 256, device=device)

    def run_loop() -> None:
        for step in range(120):
            gpu = cpu_buf.cuda(non_blocking=False)  # dominant H2D
            c = a @ b  # tiny compute
            _ = gpu.cpu()  # dominant DtoH
            if step % 4 == 0:
                _ = c.sum().item()  # secondary, sub-dominant sync
        torch.cuda.synchronize()

    def _decisive(s: dict) -> str:
        active = max(s.get("gpu_active_us", 1), 1)
        return (
            f"pcie_ratio={s.get('gpu_memcpy_time_us', 0) / active:.2f} "
            f"sync_fraction={s.get('sync_fraction', 0):.2f} "
            f"util={s.get('util', 0):.2f}"
        )

    return _run_variant(
        name="mixed_pcie_vs_sync",
        group="mixed",
        expected=Verdict.PCIE_BOUND,
        run_loop=run_loop,
        decisive_fn=_decisive,
        intended_order="pcie_ratio (~0.70) > sync_fraction (~0.30)",
    )


def plant_mixed_checkpoint_vs_dataloader() -> ExpandedResult:
    """checkpoint-dominant + dataloader present -> CHECKPOINT_BOUND.

    Intended share ordering
    ------------------------
        checkpoint fires via dtoh_count (>= 50, a SPECIFIC cause) and dataloader
        fires via dl_share (>= 0.20, the GENERIC fallback). ckpt_share and
        dl_share may be comparable — that is the point: a specific cause must
        OUTRANK the generic DataLoader fallback regardless of relative share
        (DataLoader is the outer call that contains the others).

    Why CHECKPOINT_BOUND
    --------------------
    A ``num_workers=0`` dataset sleeps 20ms/sample (real dataloader idle) while a
    30-layer model (60 tensors) is saved twice (~120 DtoH-Pageable events). In
    the engine, checkpoint is collected into the ``specific`` candidate list and
    DataLoader is only the ``elif dl_fired`` fallback that runs when ``specific``
    is empty — so checkpoint wins even if dl_share >= ckpt_share. This exercises
    the "specific outranks generic" arm of the competition with a real workload.
    """
    _require_cuda()
    import time

    import torch  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader, Dataset  # type: ignore

    class SleepDataset(Dataset):
        def __len__(self) -> int:
            return 12

        def __getitem__(self, idx: int) -> "torch.Tensor":
            time.sleep(0.020)  # real dataloader idle (the generic cause)
            return torch.randn(32, 64)

    device = torch.device("cuda")
    layers: list[torch.nn.Module] = []
    for _ in range(30):
        layers.append(torch.nn.Linear(64, 64))
        layers.append(torch.nn.ReLU())
    model = torch.nn.Sequential(*layers).to(device)
    loader = DataLoader(SleepDataset(), batch_size=4, num_workers=0)

    def run_loop() -> None:
        for batch in loader:  # produces DataLoader idle every fetch
            x = batch.to(device)
            _ = model(x).sum()
        torch.cuda.synchronize()
        fd, ckpt_path = tempfile.mkstemp(suffix=".pt", prefix="gt_expanded_ckpt_")
        os.close(fd)
        try:
            for _ in range(2):  # ~120 DtoH-Pageable events: the specific cause
                torch.save(model.state_dict(), ckpt_path)
        finally:
            try:
                os.unlink(ckpt_path)
            except OSError:
                pass

    def _decisive(s: dict) -> str:
        idle = max(s.get("idle_us", 1), 1)
        return (
            f"dtoh={s.get('checkpoint_dtoh_count', 0)} "
            f"ckpt_share={s.get('checkpoint_us', 0) / idle:.2f} "
            f"dl_share={s.get('dataloader_us', 0) / idle:.2f}"
        )

    return _run_variant(
        name="mixed_checkpoint_vs_dataloader",
        group="mixed",
        expected=Verdict.CHECKPOINT_BOUND,
        run_loop=run_loop,
        decisive_fn=_decisive,
        intended_order="checkpoint (specific) outranks dataloader (generic fallback)",
    )


# ===========================================================================
# 3. NEAR-BOUNDARY CASES
# ===========================================================================
# One workload just ABOVE a threshold (must fire) and one just BELOW (must NOT
# fire that verdict). Covered for sync (the util < 0.70 boundary behind Bug-2)
# and dataloader (the dl_share >= 0.20 boundary).


def plant_sync_boundary_above() -> ExpandedResult:
    """sync just-ABOVE firing: util ~0.60 (< 0.70) -> SYNC_BOUND.

    Construction
    ------------
    ``_synced_matmul_run_loop`` with a 2048x2048 matmul (avg_kernel_dur ~ms,
    comfortably >= 50us) and util pinned to ~0.61 by the measured post-sync sleep
    -- the side of the Bug-2 boundary where the sync rule is allowed to fire.
    sync_fraction ~1.0. Expected SYNC_BOUND.

    The FIRST T4 run used a 768x768 matmul with ``c.sum().item()`` and NO idle
    control; util landed at 0.85 (hit healthy_85) instead of the intended ~0.65.
    Pinning util with the measured sleep is what fixes that.

    This boundary IS util-sensitive: ``time.sleep`` is imprecise at the sub-ms
    scale, so if the sized sleep rounds up the achieved util lands somewhat BELOW
    the [0.55, 0.68] target -- still < 0.70, so the assertion (SYNC_BOUND) holds.
    The achieved-util column reports where it actually landed.
    """
    _require_cuda()

    # 2048x2048 -> matmul ~2ms, so the util-0.61 sleep (~1.3ms) is in time.sleep's
    # reliable range; the natural launch gap nudges achieved util a touch lower.
    run_loop = _synced_matmul_run_loop(matmul_dim=2048, target_util=0.61)

    def _decisive(s: dict) -> str:
        return (
            f"util={s.get('util', 0):.2f} "
            f"sync_fraction={s.get('sync_fraction', 0):.2f} "
            f"avg={s.get('avg_kernel_dur', 0):.0f}us"
        )

    return _run_variant(
        name="sync_boundary_above",
        group="boundary",
        expected=Verdict.SYNC_BOUND,
        run_loop=run_loop,
        decisive_fn=_decisive,
        intended_order="util ~0.60 (just below the 0.70 sync guard) -> sync fires",
    )


def plant_sync_boundary_below() -> ExpandedResult:
    """sync just-BELOW firing: util ~0.75 (>= 0.70) with sync present -> UNKNOWN.

    Construction
    ------------
    1280x1280 matmul + ``.item()`` every step. The larger matmul pushes util a
    touch ABOVE 0.70, where the sync rule's ``util < 0.70`` guard refuses to
    fire. Because sync_fraction is still ~1.0, the strengthened no_dominant gate
    (Bug-2 fix) ALSO refuses to call it HEALTHY at the 0.70 tier. The result is
    the documented "dead zone": neither SYNC_BOUND nor HEALTHY -> UNKNOWN.

    Expected UNKNOWN. WHY this is the right NON-firing outcome: the whole Bug-2
    fix was that a sync-dominated trace must NOT be called HEALTHY; at util in
    [0.70, 0.85) with sync dominating, refusing both a sync verdict (util guard)
    and a healthy verdict (no_dominant guard) is the engine being correctly
    conservative rather than confidently wrong. If util instead clears 0.85 the
    verdict is HEALTHY (healthy_85, which ignores sync_fraction) — also a NON-
    SYNC outcome, which is what this boundary asserts. Either way the assertion
    is "does NOT fire SYNC_BOUND"; the achieved-util column shows which branch.

    NOTE: this is the one variant whose expected verdict is the engine's
    conservative fallback. It is included precisely because the util-0.70 edge is
    where Bug-2 lived — pinning the post-fix behaviour here is the point.

    Construction
    ------------
    ``_synced_matmul_run_loop`` with a LARGE 4096x4096 matmul and util pinned to
    ~0.80. The large matmul is deliberate: the sized post-sync sleep is then
    several ms (well inside ``time.sleep``'s reliable range) AND the launch gap is
    negligible relative to the matmul, so achieved util lands tightly in the
    [0.70, 0.85) dead zone rather than slipping below 0.70 (which would fire
    SYNC_BOUND) or above 0.85 (HEALTHY). It also does real GPU work, so gpu_active
    is far above the pcie active-time floor -- this case previously tripped the
    near-zero-gpu-active pcie artifact (got PCIE_BOUND); with the engine's pcie
    floor fix AND real work here, that artifact cannot recur.
    """
    _require_cuda()

    # 4096x4096 -> matmul ~tens of ms, so the util-0.80 sleep is several ms
    # (reliable) and the launch gap is negligible -> util lands solidly in the
    # [0.70, 0.85) dead zone. Fewer steps keep the (heavy) run a second or two.
    run_loop = _synced_matmul_run_loop(matmul_dim=4096, target_util=0.80, n_steps=60)

    def _decisive(s: dict) -> str:
        return (
            f"util={s.get('util', 0):.2f} "
            f"sync_fraction={s.get('sync_fraction', 0):.2f} "
            f"avg={s.get('avg_kernel_dur', 0):.0f}us"
        )

    return _run_variant(
        name="sync_boundary_below",
        group="boundary",
        expected=Verdict.UNKNOWN,
        run_loop=run_loop,
        decisive_fn=_decisive,
        intended_order="util ~0.80 (>= 0.70 sync guard) + sync dominant -> dead zone",
    )


def plant_dataloader_boundary_above() -> ExpandedResult:
    """dataloader just-ABOVE firing: dl_share ~0.27 (> 0.20) -> DATALOADER_BOUND.

    Construction
    ------------
    Two idle sources whose RATIO sets dl_share precisely (robust on a T4 because
    it is governed by ``time.sleep`` durations, not GPU speed):
      * a ``num_workers=0`` dataset that sleeps 10ms/sample (dataloader idle), and
      * a bare ``time.sleep(0.027)`` in the loop body (unattributed idle — it
        matches no pattern list and is not inside a DataLoader call).
    dl_share ~= 10 / (10 + 27) ~= 0.27, just above the 0.20 firing line; util is
    tiny (the sleeps dominate wall clock) so the HEALTHY gate stays silent.
    Expected DATALOADER_BOUND.

    Real GPU work: a 1024x1024 matmul block per iter puts gpu_active well above the
    pcie active-time floor (the FIRST T4 run used a tiny 64x64 matmul -> near-zero
    gpu_active, which tripped the pcie-ratio artifact). With real work, pcie_ratio
    is ~0 and the dl_share boundary is what the verdict turns on.
    """
    _require_cuda()
    import time

    import torch  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader, Dataset  # type: ignore

    class SleepDataset(Dataset):
        def __len__(self) -> int:
            return 10

        def __getitem__(self, idx: int) -> "torch.Tensor":
            time.sleep(0.010)  # dataloader idle numerator
            return torch.randn(64, 64)

    device = torch.device("cuda")
    loader = DataLoader(SleepDataset(), batch_size=2, num_workers=0)
    cmat = torch.randn(1024, 1024, device=device)

    def run_loop() -> None:
        for batch in loader:
            x = batch.to(device)
            h = cmat
            for _ in range(4):  # real GPU work -> gpu_active >> pcie floor
                h = torch.relu(h @ cmat)
            _ = h.sum() + x.sum()
            time.sleep(0.027)  # bare, UNATTRIBUTED idle (denominator inflation)
        torch.cuda.synchronize()

    def _decisive(s: dict) -> str:
        idle = max(s.get("idle_us", 1), 1)
        return (
            f"dl_share={s.get('dataloader_us', 0) / idle:.2f} "
            f"util={s.get('util', 0):.2f}"
        )

    return _run_variant(
        name="dataloader_boundary_above",
        group="boundary",
        expected=Verdict.DATALOADER_BOUND,
        run_loop=run_loop,
        decisive_fn=_decisive,
        intended_order="dl_share ~0.27 (just above the 0.20 firing line)",
    )


def plant_dataloader_boundary_below() -> ExpandedResult:
    """dataloader just-BELOW firing: dl_share ~0.13 (< 0.20) -> UNKNOWN.

    Construction
    ------------
    Same two-sleep recipe as the above-boundary variant, but with the ratio
    flipped so the dataloader share lands under the line:
      * dataset sleeps 5ms/sample (dataloader idle), and
      * a bare ``time.sleep(0.033)`` in the loop body (unattributed idle).
    dl_share ~= 5 / (5 + 33) ~= 0.13, below the 0.20 firing line. No other cause
    is present and util is tiny (< 0.45), so nothing fires. Expected UNKNOWN.

    WHY UNKNOWN (not HEALTHY): util is far below the 0.45 borderline-healthy tier
    AND the dataloader share is below its firing threshold, so the engine has no
    signal strong enough to name — the honest answer is UNKNOWN. This pins the
    lower side of the dl_share >= 0.20 boundary.

    Real GPU work: the FIRST T4 run used a tiny 64x64 matmul, leaving gpu_active
    near zero, which tripped the near-zero-gpu-active pcie artifact and produced
    PCIE_BOUND instead of UNKNOWN. With the engine's pcie active-time floor fix
    AND a real 1024x1024 matmul block here (gpu_active well above the floor,
    pcie_ratio ~0), the verdict turns purely on dl_share < 0.20 -> UNKNOWN.
    """
    _require_cuda()
    import time

    import torch  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader, Dataset  # type: ignore

    class SleepDataset(Dataset):
        def __len__(self) -> int:
            return 10

        def __getitem__(self, idx: int) -> "torch.Tensor":
            time.sleep(0.005)  # smaller dataloader idle numerator
            return torch.randn(64, 64)

    device = torch.device("cuda")
    loader = DataLoader(SleepDataset(), batch_size=2, num_workers=0)
    cmat = torch.randn(1024, 1024, device=device)

    def run_loop() -> None:
        for batch in loader:
            x = batch.to(device)
            h = cmat
            for _ in range(4):  # real GPU work -> gpu_active >> pcie floor
                h = torch.relu(h @ cmat)
            _ = h.sum() + x.sum()
            time.sleep(0.033)  # larger bare idle -> dl_share below 0.20
        torch.cuda.synchronize()

    def _decisive(s: dict) -> str:
        idle = max(s.get("idle_us", 1), 1)
        return (
            f"dl_share={s.get('dataloader_us', 0) / idle:.2f} "
            f"util={s.get('util', 0):.2f}"
        )

    return _run_variant(
        name="dataloader_boundary_below",
        group="boundary",
        expected=Verdict.UNKNOWN,
        run_loop=run_loop,
        decisive_fn=_decisive,
        intended_order="dl_share ~0.13 (just below the 0.20 firing line)",
    )


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------
# Each entry is a zero-arg callable returning an ExpandedResult. Severity
# builders are bound to a level with functools.partial.
SEVERITY_VARIANTS: tuple[Callable[[], ExpandedResult], ...] = tuple(
    partial(builder, level)
    for builder in (
        plant_dataloader_severity,
        plant_sync_severity,
        plant_checkpoint_severity,
        plant_pcie_severity,
    )
    for level in _SEVERITY_LEVELS
)

MIXED_VARIANTS: tuple[Callable[[], ExpandedResult], ...] = (
    plant_mixed_dataloader_vs_tiny_kernels,
    plant_mixed_sync_vs_pcie,
    plant_mixed_pcie_vs_sync,
    plant_mixed_checkpoint_vs_dataloader,
)

BOUNDARY_VARIANTS: tuple[Callable[[], ExpandedResult], ...] = (
    plant_sync_boundary_above,
    plant_sync_boundary_below,
    plant_dataloader_boundary_above,
    plant_dataloader_boundary_below,
)

ALL_EXPANDED_VARIANTS: tuple[Callable[[], ExpandedResult], ...] = (
    *SEVERITY_VARIANTS,
    *MIXED_VARIANTS,
    *BOUNDARY_VARIANTS,
)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
def run_all_expanded(
    variants: tuple[Callable[[], ExpandedResult], ...] | None = None,
) -> ExpandedReport:
    """Run every variant and return an ``ExpandedReport``.

    Honest reporting contract (same as ``ground_truth.run_all``):
      * A variant that throws is recorded as a mismatch with the exception
        message in ``error`` — never retried or silently skipped.
      * The report is printed before returning so a Colab user sees the table
        even if a caller forgets to print it.
    """
    _require_cuda()
    targets = variants if variants is not None else ALL_EXPANDED_VARIANTS

    results: list[ExpandedResult] = []
    for fn in targets:
        label = getattr(fn, "func", fn).__name__
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001 — every failure is data
            log.exception("variant %s crashed", label)
            results.append(
                ExpandedResult(
                    name=label,
                    group="error",
                    expected=Verdict.UNKNOWN,
                    actual=Verdict.UNKNOWN,
                    confidence=0.0,
                    match=False,
                    decisive_metric="(crashed)",
                    intended_order="",
                    achieved_order="",
                    cause_fired=None,
                    stats={},
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    correct = sum(1 for r in results if r.match)
    mismatches = [r for r in results if not r.match]
    report = ExpandedReport(
        results=results,
        correct=correct,
        total=len(results),
        mismatches=mismatches,
    )
    _print_expanded_report(report)
    return report


def _v(verdict: Verdict) -> str:
    return verdict.value if isinstance(verdict, Verdict) else str(verdict)


def _print_expanded_report(report: ExpandedReport) -> None:
    """Grouped ASCII table: case | expected | actual | MATCH | conf | decisive.

    Plain ``print`` (no rich) so it renders identically in Colab cells, terminal
    pipes, and CI logs — same as ``ground_truth._print_report``.
    """
    print()
    print("Expanded ground-truth harness (boundaries + mixed workloads)")
    print("=" * 116)

    # Stable group order for readability.
    group_titles = [
        ("severity:dataloader", "SEVERITY SWEEP — dataloader"),
        ("severity:sync", "SEVERITY SWEEP — sync"),
        ("severity:checkpoint", "SEVERITY SWEEP — checkpoint"),
        ("severity:pcie", "SEVERITY SWEEP — pcie"),
        ("mixed", "MIXED / COMPETING (intended winner by construction)"),
        ("boundary", "NEAR-BOUNDARY (just above must fire / just below must not)"),
        ("error", "ERRORS"),
    ]
    seen_groups = {r.group for r in report.results}

    for group, title in group_titles:
        rows = [r for r in report.results if r.group == group]
        if not rows:
            continue
        print()
        print(title)
        print("-" * 116)
        print(
            f"{'case':<32} {'expected':<18} {'actual':<18} {'result':<9} "
            f"{'conf':<6} decisive"
        )
        for r in rows:
            marker = "MATCH" if r.match else "MISMATCH"
            print(
                f"{r.name:<32} {_v(r.expected):<18} {_v(r.actual):<18} {marker:<9} "
                f"{r.confidence:<6.2f} {r.decisive_metric}"
            )
            # For mixed / boundary the ORDERING is the assertion — show it.
            if r.intended_order:
                print(f"{'':<32} intended: {r.intended_order}")
                print(f"{'':<32} achieved: {r.achieved_order}")
            if r.error:
                print(f"{'':<32} error: {r.error}")

    # Any group we didn't have a title for (future-proofing) — print raw.
    for group in seen_groups - {g for g, _ in group_titles}:
        rows = [r for r in report.results if r.group == group]
        print()
        print(f"{group} (untitled group)")
        print("-" * 116)
        for r in rows:
            marker = "MATCH" if r.match else "MISMATCH"
            print(
                f"{r.name:<32} {_v(r.expected):<18} {_v(r.actual):<18} {marker:<9} "
                f"{r.confidence:<6.2f} {r.decisive_metric}"
            )

    print()
    print("=" * 116)
    print(f"accuracy: {report.correct}/{report.total}")

    if report.mismatches:
        print()
        print("Mismatches (do NOT tune the engine to silence these — investigate):")
        print(
            "  cause_fired=yes -> the workload DID hit the intended firing "
            "condition; a wrong verdict is an ENGINE bug."
        )
        print(
            "  cause_fired=no  -> the workload did NOT achieve its intended "
            "shares; fix the WORKLOAD in this file, not the engine."
        )
        for r in report.mismatches:
            if r.cause_fired is None:
                cf = "n/a (healthy/unknown target)"
            else:
                cf = "yes" if r.cause_fired else "no"
            print()
            print(
                f"  - {r.name}: expected {_v(r.expected)}, got {_v(r.actual)}  "
                f"[cause_fired={cf}]"
            )
            if r.intended_order:
                print(f"      intended order: {r.intended_order}")
            if r.achieved_order:
                print(f"      achieved order: {r.achieved_order}")
            if r.trace_path:
                print(f"      trace:  {r.trace_path}")
                print(f"      replay: gpu-doctor {r.trace_path} --explain")
            for k in (
                "util",
                "idle_us",
                "dataloader_us",
                "memcpy_us",
                "gpu_memcpy_time_us",
                "gpu_active_us",
                "checkpoint_us",
                "sync_us",
                "sync_fraction",
                "avg_kernel_dur",
                "tiny_kernel_ratio",
                "checkpoint_dtoh_count",
                "rule",
            ):
                if k in r.stats:
                    print(f"      {k}={r.stats[k]}")


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m accuracy.ground_truth_expanded``.

    Exit codes (same convention as ``accuracy.ground_truth.main``):
      0 — every variant matched its expected verdict
      1 — at least one mismatch
      2 — torch / CUDA unavailable on this host
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        report = run_all_expanded()
    except TorchUnavailable as exc:
        print(f"cannot run expanded ground-truth harness: {exc}")
        return 2
    return 0 if report.correct == report.total else 1


if __name__ == "__main__":
    sys.exit(main())
