"""Smoke test for the ground-truth accuracy harness — no GPU required.

The actual accuracy measurement happens on a Colab GPU runtime by running
``python -m accuracy.ground_truth``. CI cannot do that — there is no GPU.
What CI *can* enforce is the module's structural contract:

  * ``accuracy.ground_truth`` imports cleanly on a torch-less host
  * the eight expected planters are present and callable
  * ``run_all`` and ``TorchUnavailable`` are exposed
  * invoking a planter without torch raises ``TorchUnavailable`` (not
    a bare ``ImportError`` / ``AttributeError``) so callers can degrade
    gracefully

This test does NOT execute any planter for real — that needs CUDA.
"""

from __future__ import annotations

import sys

import pytest


def test_module_imports_without_requiring_torch() -> None:
    """The module must import on the no-torch dev/CI box."""
    import accuracy.ground_truth  # noqa: F401


def test_eight_planters_exposed() -> None:
    """Every verdict planter named in the design is present and callable."""
    from accuracy import ground_truth

    expected_names = (
        "plant_dataloader_bound",
        "plant_pcie_bound",
        "plant_kernel_launch_bound",
        "plant_checkpoint_bound",
        "plant_checkpoint_strong",
        "plant_sync_bound",
        "plant_nccl_bound",
        "plant_healthy",
    )
    for name in expected_names:
        fn = getattr(ground_truth, name, None)
        assert callable(fn), f"planter missing or not callable: {name}"

    assert len(ground_truth.ALL_PLANTERS) == 8, (
        f"ALL_PLANTERS must contain exactly 8 planters, "
        f"got {len(ground_truth.ALL_PLANTERS)}"
    )


def test_run_all_callable() -> None:
    from accuracy import ground_truth

    assert callable(ground_truth.run_all)


def test_torch_unavailable_is_a_runtime_error() -> None:
    """Mirrors the contract baked into ``gpu_doctor_agent.torch_source``."""
    from accuracy import ground_truth

    assert issubclass(ground_truth.TorchUnavailable, RuntimeError)


def test_planters_raise_torch_unavailable_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every planter must degrade to ``TorchUnavailable`` on a torch-less host.

    We monkeypatch ``_TORCH_AVAILABLE`` so the test runs identically on a
    runner that happens to have torch installed and one that does not.
    """
    from accuracy import ground_truth

    monkeypatch.setattr(ground_truth, "_TORCH_AVAILABLE", False)

    for fn in ground_truth.ALL_PLANTERS:
        if getattr(fn, "_synthetic_only", False):
            continue  # plant_nccl_bound, plant_checkpoint_strong
        with pytest.raises(ground_truth.TorchUnavailable):
            fn()


def test_run_all_raises_torch_unavailable_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_all`` itself must fail fast with a typed error, not crash mid-loop."""
    from accuracy import ground_truth

    monkeypatch.setattr(ground_truth, "_TORCH_AVAILABLE", False)
    with pytest.raises(ground_truth.TorchUnavailable):
        ground_truth.run_all()


def test_main_returns_2_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``python -m accuracy.ground_truth`` on a no-GPU host exits 2, not 1."""
    from accuracy import ground_truth

    monkeypatch.setattr(ground_truth, "_TORCH_AVAILABLE", False)
    rc = ground_truth.main([])
    assert rc == 2
    out = capsys.readouterr().out
    assert "cannot run ground-truth harness" in out


def test_no_torch_in_sys_modules_after_import_on_torchless_host() -> None:
    """Importing the harness must not have side-effect-loaded torch.

    Trivially true on a runner with torch already installed; skip there so
    the assertion stays meaningful only where it can actually catch a
    regression (the no-torch CI / dev box).
    """
    from accuracy import ground_truth

    if ground_truth._TORCH_AVAILABLE:
        pytest.skip("torch present on this runner — invariant trivially holds")

    assert "torch" not in sys.modules


def test_planter_result_shape_is_complete() -> None:
    """The result dataclass exposes every field the report consumes.

    Locking the shape so future refactors don't quietly drop a field that
    ``_print_report`` (or downstream tooling) depends on.
    """
    from accuracy.ground_truth import PlanterResult

    required = {
        "name",
        "expected",
        "actual",
        "confidence",
        "match",
        "decisive_metric",
        "stats",
        "trace_path",
        "error",
    }
    assert required.issubset(PlanterResult.__dataclass_fields__.keys())


def test_accuracy_report_shape_is_complete() -> None:
    from accuracy.ground_truth import AccuracyReport

    required = {"results", "correct", "total", "mismatches"}
    assert required.issubset(AccuracyReport.__dataclass_fields__.keys())
