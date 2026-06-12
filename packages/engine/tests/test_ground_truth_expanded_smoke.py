"""Smoke test for the EXPANDED ground-truth harness — no GPU required.

The actual accuracy measurement happens on a Colab GPU runtime by running
``python -m accuracy.ground_truth_expanded``. CI has no GPU. What CI *can*
enforce is the module's structural contract:

  * ``accuracy.ground_truth_expanded`` imports cleanly on a torch-less host
  * the severity / mixed / boundary builders and registries are present
  * ``run_all_expanded`` and ``TorchUnavailable`` are exposed
  * invoking a variant without torch raises ``TorchUnavailable`` (not a bare
    ``ImportError`` / ``AttributeError``) so callers can degrade gracefully

This test does NOT execute any variant for real — that needs CUDA. It mirrors
``tests/test_ground_truth_smoke.py`` for the single-cause harness.
"""

from __future__ import annotations

import sys

import pytest


def test_module_imports_without_requiring_torch() -> None:
    """The module must import on the no-torch dev/CI box."""
    import accuracy.ground_truth_expanded  # noqa: F401


def test_severity_builders_exposed() -> None:
    """Every parametrized severity builder is present and callable."""
    from accuracy import ground_truth_expanded as gte

    for name in (
        "plant_dataloader_severity",
        "plant_sync_severity",
        "plant_checkpoint_severity",
        "plant_pcie_severity",
    ):
        fn = getattr(gte, name, None)
        assert callable(fn), f"severity builder missing or not callable: {name}"


def test_mixed_and_boundary_builders_exposed() -> None:
    """Every mixed / boundary builder named in the design is present and callable."""
    from accuracy import ground_truth_expanded as gte

    for name in (
        # mixed / competing
        "plant_mixed_dataloader_vs_tiny_kernels",
        "plant_mixed_sync_vs_pcie",
        "plant_mixed_pcie_vs_sync",
        "plant_mixed_checkpoint_vs_dataloader",
        # near-boundary
        "plant_sync_boundary_above",
        "plant_sync_boundary_below",
        "plant_dataloader_boundary_above",
        "plant_dataloader_boundary_below",
    ):
        fn = getattr(gte, name, None)
        assert callable(fn), f"builder missing or not callable: {name}"


def test_registries_have_expected_sizes() -> None:
    """The variant registries match the documented coverage counts."""
    from accuracy import ground_truth_expanded as gte

    # 4 causes x 3 severities = 12.
    assert len(gte.SEVERITY_VARIANTS) == 12, (
        f"SEVERITY_VARIANTS must contain 4 causes x 3 levels = 12, "
        f"got {len(gte.SEVERITY_VARIANTS)}"
    )
    assert len(gte.MIXED_VARIANTS) == 4, (
        f"MIXED_VARIANTS must contain 4 competing workloads, "
        f"got {len(gte.MIXED_VARIANTS)}"
    )
    assert len(gte.BOUNDARY_VARIANTS) == 4, (
        f"BOUNDARY_VARIANTS must contain 4 near-boundary workloads, "
        f"got {len(gte.BOUNDARY_VARIANTS)}"
    )
    assert len(gte.ALL_EXPANDED_VARIANTS) == 20, (
        f"ALL_EXPANDED_VARIANTS must aggregate to 12 + 4 + 4 = 20, "
        f"got {len(gte.ALL_EXPANDED_VARIANTS)}"
    )
    # Registry entries must be callable (zero-arg) variant runners.
    for fn in gte.ALL_EXPANDED_VARIANTS:
        assert callable(fn)


def test_run_all_expanded_callable() -> None:
    from accuracy import ground_truth_expanded as gte

    assert callable(gte.run_all_expanded)


def test_torch_unavailable_is_a_runtime_error() -> None:
    """Mirrors the contract baked into ``gpu_doctor_agent.torch_source``."""
    from accuracy import ground_truth_expanded as gte

    assert issubclass(gte.TorchUnavailable, RuntimeError)


def test_severity_levels_constant() -> None:
    """The three documented severity levels are exposed for builders to bind."""
    from accuracy import ground_truth_expanded as gte

    assert gte._SEVERITY_LEVELS == ("mild", "moderate", "severe")


def test_variants_raise_torch_unavailable_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every registered variant must degrade to ``TorchUnavailable`` torch-less.

    We monkeypatch ``_TORCH_AVAILABLE`` so the test runs identically on a runner
    that happens to have torch installed and one that does not.
    """
    from accuracy import ground_truth_expanded as gte

    monkeypatch.setattr(gte, "_TORCH_AVAILABLE", False)

    for fn in gte.ALL_EXPANDED_VARIANTS:
        with pytest.raises(gte.TorchUnavailable):
            fn()


def test_severity_builders_raise_torch_unavailable_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling a severity builder directly (any level) degrades cleanly too."""
    from accuracy import ground_truth_expanded as gte

    monkeypatch.setattr(gte, "_TORCH_AVAILABLE", False)

    builders = (
        gte.plant_dataloader_severity,
        gte.plant_sync_severity,
        gte.plant_checkpoint_severity,
        gte.plant_pcie_severity,
    )
    for builder in builders:
        for level in gte._SEVERITY_LEVELS:
            with pytest.raises(gte.TorchUnavailable):
                builder(level)


def test_run_all_expanded_raises_torch_unavailable_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_all_expanded`` must fail fast with a typed error, not crash mid-loop."""
    from accuracy import ground_truth_expanded as gte

    monkeypatch.setattr(gte, "_TORCH_AVAILABLE", False)
    with pytest.raises(gte.TorchUnavailable):
        gte.run_all_expanded()


def test_main_returns_2_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``python -m accuracy.ground_truth_expanded`` on a no-GPU host exits 2."""
    from accuracy import ground_truth_expanded as gte

    monkeypatch.setattr(gte, "_TORCH_AVAILABLE", False)
    rc = gte.main([])
    assert rc == 2
    out = capsys.readouterr().out
    assert "cannot run expanded ground-truth harness" in out


def test_no_torch_in_sys_modules_after_import_on_torchless_host() -> None:
    """Importing the harness must not have side-effect-loaded torch.

    Trivially true on a runner with torch already installed; skip there so the
    assertion stays meaningful only where it can catch a regression.
    """
    from accuracy import ground_truth_expanded as gte

    if gte._TORCH_AVAILABLE:
        pytest.skip("torch present on this runner — invariant trivially holds")

    assert "torch" not in sys.modules


def test_expanded_result_shape_is_complete() -> None:
    """The result dataclass exposes every field the report consumes."""
    from accuracy.ground_truth_expanded import ExpandedResult

    required = {
        "name",
        "group",
        "expected",
        "actual",
        "confidence",
        "match",
        "decisive_metric",
        "intended_order",
        "achieved_order",
        "cause_fired",
        "stats",
        "trace_path",
        "error",
    }
    assert required.issubset(ExpandedResult.__dataclass_fields__.keys())


def test_expanded_report_shape_is_complete() -> None:
    from accuracy.ground_truth_expanded import ExpandedReport

    required = {"results", "correct", "total", "mismatches"}
    assert required.issubset(ExpandedReport.__dataclass_fields__.keys())


def test_share_helpers_pure_python_no_torch() -> None:
    """The share / cause-fired helpers are pure-Python and need no GPU.

    They are what makes a mismatch diagnosable from the printed table, so they
    must work on the no-torch box. Exercise them on a hand-built stats dict.
    """
    from accuracy.ground_truth_expanded import (
        _expected_cause_fired,
        _share_ordering,
        _shares,
    )
    from gpu_doctor_engine.types import Verdict

    stats = {
        "idle_us": 1000,
        "gpu_active_us": 500,
        "dataloader_us": 800,
        "sync_fraction": 0.10,
        "checkpoint_us": 0,
        "nccl_us": 0,
        "gpu_memcpy_time_us": 50,
        "util": 0.30,
        "avg_kernel_dur": 80.0,
        "memcpy_us": 50,
        "checkpoint_dtoh_count": 0,
    }
    shares = _shares(stats)
    assert shares["dl_share"] == pytest.approx(0.80)
    assert shares["pcie_ratio"] == pytest.approx(0.10)
    # dl_share is the largest, so it must lead the ordering string.
    assert _share_ordering(stats).startswith("dl_share=0.80")

    # dataloader fired (dl_share 0.80 >= 0.20); pcie did not (ratio 0.10 < 0.50).
    assert _expected_cause_fired(Verdict.DATALOADER_BOUND, stats) is True
    assert _expected_cause_fired(Verdict.PCIE_BOUND, stats) is False
    # HEALTHY / UNKNOWN have no single cause to fire.
    assert _expected_cause_fired(Verdict.HEALTHY, stats) is None
    assert _expected_cause_fired(Verdict.UNKNOWN, stats) is None


def test_print_report_runs_without_torch(capsys: pytest.CaptureFixture[str]) -> None:
    """``_print_expanded_report`` must render a hand-built report with no GPU.

    Guards the report formatter (grouping, ordering lines, mismatch block)
    against regressions on the CI box, independent of any real workload.
    """
    from accuracy.ground_truth_expanded import (
        ExpandedReport,
        ExpandedResult,
        _print_expanded_report,
    )
    from gpu_doctor_engine.types import Verdict

    match = ExpandedResult(
        name="sync_severe",
        group="severity:sync",
        expected=Verdict.SYNC_BOUND,
        actual=Verdict.SYNC_BOUND,
        confidence=0.9,
        match=True,
        decisive_metric="sync_fraction=0.95 util=0.30",
        intended_order="",
        achieved_order="sync_fraction=0.95 > dl_share=0.00",
        cause_fired=True,
        stats={"util": 0.30, "sync_fraction": 0.95},
    )
    mismatch = ExpandedResult(
        name="mixed_sync_vs_pcie",
        group="mixed",
        expected=Verdict.SYNC_BOUND,
        actual=Verdict.PCIE_BOUND,
        confidence=0.7,
        match=False,
        decisive_metric="sync_fraction=0.30 pcie_ratio=0.55",
        intended_order="sync_fraction (~0.55) > pcie_ratio (~0.30)",
        achieved_order="pcie_ratio=0.55 > sync_fraction=0.30",
        cause_fired=False,  # workload didn't achieve the intended ordering
        stats={"util": 0.40, "sync_fraction": 0.30, "rule": "pcie_ratio_50"},
        trace_path="/tmp/example.json",
    )
    report = ExpandedReport(
        results=[match, mismatch], correct=1, total=2, mismatches=[mismatch]
    )
    _print_expanded_report(report)

    out = capsys.readouterr().out
    assert "accuracy: 1/2" in out
    assert "MATCH" in out and "MISMATCH" in out
    assert "intended:" in out and "achieved:" in out
    assert "cause_fired=no" in out  # the construction-vs-engine discriminator
    assert "replay: gpu-doctor /tmp/example.json --explain" in out
