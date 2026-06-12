"""Tests for recalibrated confidence_from_share tiers."""

from __future__ import annotations

import pytest

from gpu_doctor_engine.detectors.confidence import confidence_from_share


@pytest.mark.parametrize(
    "share,expected",
    [
        (0.95, 0.95),
        (0.90, 0.95),
        (0.80, 0.90),
        (0.75, 0.90),
        (0.65, 0.82),
        (0.60, 0.82),
        (0.50, 0.72),
        (0.40, 0.72),
        (0.30, 0.60),
        (0.25, 0.60),
        (0.20, 0.60),
    ],
)
def test_confidence_tiers(share: float, expected: float) -> None:
    assert confidence_from_share(share) == pytest.approx(expected)


def test_healthy_diagnosis_has_no_confidence(healthy_trace) -> None:
    from gpu_doctor_engine import diagnose

    diag = diagnose(healthy_trace)
    assert diag.confidence is None
