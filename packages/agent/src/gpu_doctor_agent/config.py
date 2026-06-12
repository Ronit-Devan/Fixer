"""Agent configuration. Immutable; built once at startup, passed by value."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence


class ConfigError(ValueError):
    """Raised when AgentConfig values violate invariants."""


def _parse_indices(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or raw.strip() == "" or raw.strip().lower() == "all":
        return None
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
        except ValueError as e:
            raise ConfigError(f"GPU_DOCTOR_GPU_INDICES: bad index {part!r}") from e
        if idx < 0:
            raise ConfigError(f"GPU_DOCTOR_GPU_INDICES: negative index {idx}")
        out.append(idx)
    return tuple(out) if out else None


def _parse_float(name: str, raw: str | None, default: float) -> float:
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ConfigError(f"{name}: not a float: {raw!r}") from e


def _parse_int(name: str, raw: str | None, default: int) -> int:
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ConfigError(f"{name}: not an int: {raw!r}") from e


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for the sampling spine.

    All time values are seconds (monotonic). Utilization thresholds are
    fractions in [0, 1] (NVML's 0..100 ints are normalized at the sampler).
    """

    sample_interval_s: float = 1.0
    gpu_indices: tuple[int, ...] | None = None
    idle_util_threshold: float = 0.20
    idle_sustain_s: float = 5.0
    recovery_util_threshold: float = 0.40
    ring_capacity: int = 600

    def __post_init__(self) -> None:
        if self.sample_interval_s <= 0:
            raise ConfigError(
                f"sample_interval_s must be > 0, got {self.sample_interval_s}"
            )
        if not (0.0 <= self.idle_util_threshold <= 1.0):
            raise ConfigError(
                f"idle_util_threshold must be in [0,1], got {self.idle_util_threshold}"
            )
        if not (0.0 <= self.recovery_util_threshold <= 1.0):
            raise ConfigError(
                f"recovery_util_threshold must be in [0,1], got {self.recovery_util_threshold}"
            )
        # Hysteresis invariant: recovery threshold must be strictly above
        # entry threshold, otherwise the detector can flap on jitter alone.
        if self.recovery_util_threshold <= self.idle_util_threshold:
            raise ConfigError(
                "recovery_util_threshold must be > idle_util_threshold "
                f"({self.recovery_util_threshold} <= {self.idle_util_threshold})"
            )
        if self.idle_sustain_s < 0:
            raise ConfigError(f"idle_sustain_s must be >= 0, got {self.idle_sustain_s}")
        if self.ring_capacity <= 0:
            raise ConfigError(f"ring_capacity must be > 0, got {self.ring_capacity}")
        if self.gpu_indices is not None:
            if any(i < 0 for i in self.gpu_indices):
                raise ConfigError(f"gpu_indices must be non-negative, got {self.gpu_indices}")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "AgentConfig":
        e = env if env is not None else os.environ
        return cls(
            sample_interval_s=_parse_float(
                "GPU_DOCTOR_SAMPLE_INTERVAL_S",
                e.get("GPU_DOCTOR_SAMPLE_INTERVAL_S"),
                1.0,
            ),
            gpu_indices=_parse_indices(e.get("GPU_DOCTOR_GPU_INDICES")),
            idle_util_threshold=_parse_float(
                "GPU_DOCTOR_IDLE_UTIL_THRESHOLD",
                e.get("GPU_DOCTOR_IDLE_UTIL_THRESHOLD"),
                0.20,
            ),
            idle_sustain_s=_parse_float(
                "GPU_DOCTOR_IDLE_SUSTAIN_S",
                e.get("GPU_DOCTOR_IDLE_SUSTAIN_S"),
                5.0,
            ),
            recovery_util_threshold=_parse_float(
                "GPU_DOCTOR_RECOVERY_UTIL_THRESHOLD",
                e.get("GPU_DOCTOR_RECOVERY_UTIL_THRESHOLD"),
                0.40,
            ),
            ring_capacity=_parse_int(
                "GPU_DOCTOR_RING_CAPACITY",
                e.get("GPU_DOCTOR_RING_CAPACITY"),
                600,
            ),
        )

    def with_overrides(
        self,
        *,
        sample_interval_s: float | None = None,
        gpu_indices: Sequence[int] | None = None,
        idle_sustain_s: float | None = None,
    ) -> "AgentConfig":
        """Return a new config with selected fields overridden (for CLI flags)."""
        return AgentConfig(
            sample_interval_s=(
                sample_interval_s if sample_interval_s is not None else self.sample_interval_s
            ),
            gpu_indices=(
                tuple(gpu_indices) if gpu_indices is not None else self.gpu_indices
            ),
            idle_util_threshold=self.idle_util_threshold,
            idle_sustain_s=(
                idle_sustain_s if idle_sustain_s is not None else self.idle_sustain_s
            ),
            recovery_util_threshold=self.recovery_util_threshold,
            ring_capacity=self.ring_capacity,
        )


# Re-export field for downstream tooling that wants to introspect defaults.
__all__ = ["AgentConfig", "ConfigError", "field"]
