"""Unit tests for the overhead measurement harness (NO torch required).

The pure stats helpers (mean/stddev/overhead_pct) are exhaustively
exercised on hand-picked inputs. ``measure_overhead`` itself requires
torch to run a real workload, so on the no-torch dev box / CI we only
assert it raises the documented ``TorchUnavailable`` error — the actual
overhead-% number is produced from Colab.
"""

from __future__ import annotations

import math

import pytest

from gpu_doctor_agent import overhead as overhead_mod
from gpu_doctor_agent import torch_source as ts_mod
from gpu_doctor_agent.overhead import (
    mean,
    measure_overhead,
    overhead_pct,
    stddev,
)
from gpu_doctor_agent.torch_source import TorchUnavailable


# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------


def test_mean_basic() -> None:
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_mean_single_value_is_identity() -> None:
    assert mean([7.5]) == pytest.approx(7.5)


def test_mean_empty_is_zero() -> None:
    """Empty sequence is the only documented zero return; do not raise."""
    assert mean([]) == 0.0


def test_mean_negative_values() -> None:
    assert mean([-1.0, 1.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# stddev
# ---------------------------------------------------------------------------


def test_stddev_population_form_on_known_inputs() -> None:
    # Population stddev of [2, 4, 4, 4, 5, 5, 7, 9] is 2.0 exactly.
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert stddev(values) == pytest.approx(2.0)


def test_stddev_zero_for_constant_sequence() -> None:
    assert stddev([3.0, 3.0, 3.0, 3.0]) == 0.0


def test_stddev_zero_for_short_sequences() -> None:
    """Documented behaviour: <2 values means "no spread", returns 0.0."""
    assert stddev([]) == 0.0
    assert stddev([42.0]) == 0.0


def test_stddev_two_values() -> None:
    # Population stddev of [0, 2]: variance = ((0-1)^2 + (2-1)^2)/2 = 1, sqrt -> 1.
    assert stddev([0.0, 2.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# overhead_pct
# ---------------------------------------------------------------------------


def test_overhead_pct_baseline_zero_returns_zero() -> None:
    """Documented edge case: baseline=0 gives 0% (no division)."""
    assert overhead_pct(0.0, 1.0) == 0.0


def test_overhead_pct_baseline_negative_returns_zero() -> None:
    assert overhead_pct(-1.0, 1.0) == 0.0


def test_overhead_pct_ten_percent() -> None:
    assert overhead_pct(1.0, 1.1) == pytest.approx(10.0)


def test_overhead_pct_negative_when_instrumented_faster() -> None:
    """A faster instrumented run reports a negative overhead — not clamped."""
    pct = overhead_pct(1.0, 0.9)
    assert pct == pytest.approx(-10.0)
    assert pct < 0


def test_overhead_pct_zero_when_identical() -> None:
    assert overhead_pct(0.5, 0.5) == pytest.approx(0.0)


def test_overhead_pct_finite_for_extreme_inputs() -> None:
    """Sanity: huge ratios stay finite (not inf / NaN) for downstream JSON."""
    pct = overhead_pct(1e-9, 1.0)
    assert math.isfinite(pct)


# ---------------------------------------------------------------------------
# measure_overhead — torch-less path
# ---------------------------------------------------------------------------


def test_measure_overhead_raises_torch_unavailable_without_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-torch host: must surface the documented error, not crash deeper."""
    monkeypatch.setattr(ts_mod, "_TORCH_AVAILABLE", False)
    with pytest.raises(TorchUnavailable):
        measure_overhead(lambda: None, repeats=2)


def test_measure_overhead_rejects_zero_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero / negative repeats is a programmer error — validate before torch check."""
    # Pretend torch is present so the repeats check is what raises.
    monkeypatch.setattr(ts_mod, "_TORCH_AVAILABLE", True)
    with pytest.raises(ValueError):
        measure_overhead(lambda: None, repeats=0)
    with pytest.raises(ValueError):
        measure_overhead(lambda: None, repeats=-3)


# ---------------------------------------------------------------------------
# Re-exports / module API
# ---------------------------------------------------------------------------


def test_overhead_module_public_api() -> None:
    """The four documented names are exported and callable."""
    for name in ("mean", "stddev", "overhead_pct", "measure_overhead"):
        assert hasattr(overhead_mod, name), f"missing public symbol: {name}"
