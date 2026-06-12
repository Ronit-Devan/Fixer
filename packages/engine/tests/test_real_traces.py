"""Parametrized integration tests against the four real PyTorch Profiler fixtures.

Each fixture was captured from a real Colab training run and has a verified
ground-truth verdict.  These tests guard against regressions in the diagnostic
rules whenever the engine logic changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_doctor_engine import diagnose, diagnose_with_stats, load_trace
from gpu_doctor_engine.types import Verdict

FIXTURES = Path(__file__).parent.parent.parent.parent / "fixtures"


@pytest.mark.parametrize(
    "filename,expected_verdict",
    [
        ("dataloader_starved.json", Verdict.DATALOADER_BOUND),
        ("pcie_bound.json", Verdict.PCIE_BOUND),
        ("kernel_launch_bound.json", Verdict.KERNEL_LAUNCH_BOUND),
        ("healthy.json", Verdict.HEALTHY),
        ("checkpoint_bound.json", Verdict.CHECKPOINT_BOUND),
        # Edge-case fixtures: regression tests for three calibration bugs fixed in v0.3.
        # Bug 1: CHECKPOINT_BOUND false-fired on tiny traces due to overhead patterns.
        ("edge_tiny_trace.json", Verdict.UNKNOWN),
        # Bug 2: CUDA-only profiles returned UNKNOWN at 2% util (wall-clock bias).
        ("edge_gpu_only.json", Verdict.HEALTHY),
        # Bug 3: borderline healthy trace (50% util, no torch.save) returned CHECKPOINT_BOUND
        # due to the same false overhead patterns. Must NOT be CHECKPOINT_BOUND.
        ("edge_borderline_healthy.json", Verdict.HEALTHY),
        # Bug 4: trace with lazy-import overhead but no real torch.save overlap.
        ("cuda_sync_stalls.json", Verdict.HEALTHY),
        # S4 guard: util is 98% so the Stage-1 HEALTHY gate claims this before
        # Stage 2 runs (despite memcpy/checkpoint candidates passing their own
        # thresholds). The dominant-cause ranking fix must not leak past the
        # HEALTHY gate.
        ("dataloader_bound.json", Verdict.HEALTHY),
    ],
)
def test_real_trace_verdict(filename: str, expected_verdict: Verdict) -> None:
    """Each real fixture must produce the ground-truth verdict with confidence > 0.5.

    The fixture path is resolved relative to the repo root so the test works
    regardless of where pytest is invoked from.
    """
    trace_path = FIXTURES / filename
    assert trace_path.exists(), f"Fixture missing: {trace_path}"

    trace = load_trace(trace_path)
    diagnosis = diagnose(trace)

    conf_repr = "n/a" if diagnosis.confidence is None else f"{diagnosis.confidence:.2f}"
    assert diagnosis.verdict == expected_verdict, (
        f"Expected {expected_verdict} for {filename}, "
        f"got {diagnosis.verdict} "
        f"(util={diagnosis.metrics.get('gpu_util', 0):.0%}, conf={conf_repr})"
    )
    if expected_verdict == Verdict.HEALTHY:
        assert diagnosis.confidence is None
    elif expected_verdict != Verdict.UNKNOWN:
        assert diagnosis.confidence is not None
        assert 0.55 <= diagnosis.confidence <= 0.98, (
            f"Confidence out of range for {filename}: {diagnosis.confidence}"
        )


def test_cuda_sync_stalls_v4_is_sync_bound() -> None:
    """Full cuda_sync_stalls_v4 trace fires SYNC_BOUND via idle-after-sync attribution.

    The trace has 401 sync events (cudaDeviceSynchronize, cudaStreamSynchronize,
    aten::item, aten::_local_scalar_dense) that produce post-sync dispatch gaps
    totalling >25% of GPU idle time.  Average kernel duration is ~132µs so the
    avg_kernel_dur >= 50 guard does not block the rule.
    """
    trace_path = FIXTURES / "cuda_sync_stalls_v4.json"
    assert trace_path.exists(), f"Fixture missing: {trace_path}"

    trace = load_trace(trace_path)
    diagnosis = diagnose(trace)

    assert diagnosis.verdict == Verdict.SYNC_BOUND, (
        f"Expected SYNC_BOUND for cuda_sync_stalls_v4.json, "
        f"got {diagnosis.verdict} "
        f"(util={diagnosis.metrics.get('gpu_util', 0):.0%}, "
        f"conf={diagnosis.confidence:.2f})"
    )
    assert diagnosis.confidence is not None
    assert 0.55 <= diagnosis.confidence <= 0.98


def test_dataloader_starved_hol_stats() -> None:
    """The dataloader_starved fixture must report HoL stats in its metrics.

    We don't assert on hol_blocking_likely because the Colab trace could go
    either way depending on sample variance; we only verify the keys are present.
    """
    trace_path = FIXTURES / "dataloader_starved.json"
    assert trace_path.exists(), f"Fixture missing: {trace_path}"

    trace = load_trace(trace_path)
    diagnosis = diagnose(trace)

    assert diagnosis.verdict == Verdict.DATALOADER_BOUND
    for key in (
        "hol_median_us",
        "hol_p99_us",
        "hol_ratio",
        "hol_blocking_likely",
        "hol_sample_count",
    ):
        assert key in diagnosis.metrics, f"Missing HoL metric key: {key}"


def test_s4_pcie_vs_sync_sync_wins() -> None:
    """S4 regression: a trace that is BOTH transfer-heavy and sync-heavy.

    pcie_ratio_50 fires (memcpy is 83% of a tiny GPU-active window) AND sync_25
    fires (sync stalls cover 70% of a huge GPU-idle window). Pre-fix, Stage-2
    ranked by share and PCIE_BOUND won because 0.83 (share-of-active) beat 0.70
    (share-of-idle) — incommensurable denominators. Ranking by the absolute
    attributed GPU-µs (sync_us≈700ms >> memcpy_time≈2ms) correctly returns
    SYNC_BOUND. This must be a genuine 2-way sort, so we assert both candidates
    passed their gates (otherwise the test degenerates to a lone winner).
    """
    trace_path = FIXTURES / "pcie_vs_sync_sync_wins.json"
    assert trace_path.exists(), f"Fixture missing: {trace_path}"

    diag, stats = diagnose_with_stats(load_trace(trace_path))
    passed = {d["rule"] for d in stats["decisions"] if d["passed"]}
    assert {"pcie_ratio_50", "sync_25"} <= passed, (
        f"expected both pcie_ratio_50 and sync_25 to fire (real 2-way sort), "
        f"got passed={passed}"
    )
    assert diag.verdict == Verdict.SYNC_BOUND, (
        f"Expected SYNC_BOUND (sync_us >> memcpy_us under abs-µs ranking), "
        f"got {diag.verdict} (rule={stats.get('rule')})"
    )
    assert diag.confidence is not None and 0.55 <= diag.confidence <= 0.98


def test_s4_pcie_vs_idle_pcie_wins() -> None:
    """S4 over-correction guard: a genuinely transfer-bound trace.

    ~500ms of memcpy dominates a large GPU-active window (pcie_ratio 0.83) while
    a WEAK sync cause still fires (30% of a small ~40ms idle window). Ranking by
    absolute attributed GPU-µs keeps PCIE_BOUND (memcpy_us≈500ms >> sync_us≈12ms)
    — the fix must not blindly hand every contested trace to the idle cause.
    Assert both candidates passed their gates so this is a real competition.
    """
    trace_path = FIXTURES / "pcie_vs_idle_pcie_wins.json"
    assert trace_path.exists(), f"Fixture missing: {trace_path}"

    diag, stats = diagnose_with_stats(load_trace(trace_path))
    passed = {d["rule"] for d in stats["decisions"] if d["passed"]}
    assert {"pcie_ratio_50", "sync_25"} <= passed, (
        f"expected both pcie_ratio_50 and sync_25 to fire (real 2-way sort), "
        f"got passed={passed}"
    )
    assert diag.verdict == Verdict.PCIE_BOUND, (
        f"Expected PCIE_BOUND (memcpy_us >> sync_us under abs-µs ranking), "
        f"got {diag.verdict} (rule={stats.get('rule')})"
    )
    assert diag.confidence is not None and 0.55 <= diag.confidence <= 0.98
