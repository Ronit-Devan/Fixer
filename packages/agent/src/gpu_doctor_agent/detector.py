"""Per-GPU idle-detection FSM.

States: HEALTHY -> IDLE_SUSPECTED -> IDLE_CONFIRMED
                       ^                  |
                       |                  v
                       +------- (recovery) HEALTHY

The detector is edge-triggered: it emits exactly one IdleEvent on the
HEALTHY/SUSPECTED -> IDLE_CONFIRMED transition, then stays silent until the
GPU recovers above `recovery_util_threshold` and dips below
`idle_util_threshold` again. The asymmetric thresholds give hysteresis so
utilization that jitters around the entry threshold doesn't flap.

Time is injected via `now_fn` to keep tests synchronous.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass
from typing import Callable

from gpu_doctor_agent.config import AgentConfig

log = logging.getLogger(__name__)


class IdleState(str, enum.Enum):
    HEALTHY = "HEALTHY"
    IDLE_SUSPECTED = "IDLE_SUSPECTED"
    IDLE_CONFIRMED = "IDLE_CONFIRMED"


@dataclass(frozen=True)
class IdleEvent:
    gpu_index: int
    started_at_s: float  # monotonic timestamp when util first dipped (suspected->)
    mean_util: float  # mean util in the confirmation window


class IdleDetector:
    """Per-GPU FSM. One instance per gpu_index."""

    def __init__(
        self,
        gpu_index: int,
        config: AgentConfig,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._gpu_index = gpu_index
        self._idle_threshold = config.idle_util_threshold
        self._recovery_threshold = config.recovery_util_threshold
        self._sustain_s = config.idle_sustain_s
        self._now_fn = now_fn
        self._state: IdleState = IdleState.HEALTHY
        self._suspected_since_s: float | None = None

    @property
    def state(self) -> IdleState:
        return self._state

    @property
    def gpu_index(self) -> int:
        return self._gpu_index

    def observe(self, mean_util: float | None, now: float | None = None) -> IdleEvent | None:
        """Feed one observation. Returns an IdleEvent on the confirming edge, else None.

        `mean_util` of None means the sampler couldn't read this GPU on this tick.
        We treat that as "no information" and hold the current state — neither
        promoting to suspected nor recovering. This avoids false confirmations
        from a transient NVML hiccup.
        """
        if mean_util is None:
            return None

        ts = now if now is not None else self._now_fn()

        if self._state is IdleState.HEALTHY:
            if mean_util < self._idle_threshold:
                self._state = IdleState.IDLE_SUSPECTED
                self._suspected_since_s = ts
                # Fall through to the SUSPECTED check on this same observation:
                # with sustain_s == 0 this confirms immediately on the first dip,
                # which is the only intuitive meaning of "sustain zero".
            else:
                return None

        if self._state is IdleState.IDLE_SUSPECTED:
            # Recovery before sustain elapses: brief dip, ignore.
            if mean_util >= self._idle_threshold:
                self._state = IdleState.HEALTHY
                self._suspected_since_s = None
                return None
            # Still idle: have we held long enough?
            assert self._suspected_since_s is not None
            if ts - self._suspected_since_s >= self._sustain_s:
                started = self._suspected_since_s
                self._state = IdleState.IDLE_CONFIRMED
                return IdleEvent(
                    gpu_index=self._gpu_index,
                    started_at_s=started,
                    mean_util=mean_util,
                )
            return None

        # IDLE_CONFIRMED: recover only when crossing the higher recovery
        # threshold (hysteresis prevents flapping on jitter near the entry
        # threshold).
        if self._state is IdleState.IDLE_CONFIRMED:
            if mean_util >= self._recovery_threshold:
                self._state = IdleState.HEALTHY
                self._suspected_since_s = None
            return None

        # Unreachable, but keeps mypy/exhaustiveness honest.
        return None
