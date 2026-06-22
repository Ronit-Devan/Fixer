"""Shared test fixtures: a Diagnosis-like fake and small builders.

These mirror the two products' Diagnosis shapes without importing them, proving
the layer really is decoupled (the tests construct their own duck types).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Verdict:
    value: str


@dataclass
class FakeDiagnosis:
    """Quacks like both et_monitor.Diagnosis and gpu_doctor_engine.Diagnosis."""

    verdict_value: str
    metrics: dict = field(default_factory=dict)
    severity: str = "warn"
    summary: str = "fake diagnosis"

    @property
    def verdict(self) -> _Verdict:
        return _Verdict(self.verdict_value)


def diag(verdict_value: str, *, metrics: dict | None = None, severity: str = "warn") -> FakeDiagnosis:
    return FakeDiagnosis(verdict_value, metrics or {}, severity)
