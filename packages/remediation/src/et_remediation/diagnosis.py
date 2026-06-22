"""Decoupling boundary between the diagnosers and the remediation layer.

The remediation layer consumes a *diagnosis* but must not import either
product's ``Diagnosis`` class (that would couple the packages and invert the
dependency). Instead it accepts anything that quacks like one — the duck-typed
``DiagnosisLike`` Protocol — and normalizes it to a small internal record.

Both ``gpu_doctor_engine.Diagnosis`` and ``et_monitor.Diagnosis`` already
satisfy this: each has ``.verdict`` (a ``str``-Enum) and ``.metrics``. The
monitor additionally has ``.severity``; the engine does not, so severity is
optional and defaults to ``""``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class DiagnosisLike(Protocol):
    """Structural type both products' Diagnosis already satisfies."""

    @property
    def verdict(self) -> object: ...  # a str-Enum; we read .value

    metrics: dict


# Which product produced a diagnosis decides which verdict->RootCause map to use.
Source = str  # "monitor" | "engine"


@dataclass(frozen=True)
class NormalizedDiagnosis:
    """Flattened, source-tagged view the remediation layer reasons over."""

    source: Source
    verdict_value: str
    severity: str
    summary: str
    metrics: dict = field(default_factory=dict)
    # True when the verdict came from a TREND projection (the problem hasn't
    # landed yet). Some pre-emptive fixes are unsafe on a prediction (e.g.
    # raising power on an imminent HEAT throttle would worsen it), so the manager
    # treats certain predicted causes as advise-only.
    predicted: bool = False


def _verdict_value(diag: object) -> str:
    """Extract the string verdict value from a Diagnosis-like object."""
    v = getattr(diag, "verdict", None)
    if v is None:
        return "unknown"
    # str-Enum -> .value; plain string -> itself.
    return getattr(v, "value", v) if not isinstance(v, str) else v


def normalize(diag: DiagnosisLike, source: Source) -> NormalizedDiagnosis:
    """Adapt either product's Diagnosis into a NormalizedDiagnosis."""
    return NormalizedDiagnosis(
        source=source,
        verdict_value=_verdict_value(diag),
        severity=str(getattr(diag, "severity", "") or ""),
        summary=str(getattr(diag, "summary", "") or ""),
        metrics=dict(getattr(diag, "metrics", {}) or {}),
        predicted=bool(getattr(diag, "predicted", False)),
    )
