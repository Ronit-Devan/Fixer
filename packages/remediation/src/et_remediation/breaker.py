"""Per-node circuit breaker governing auto-apply.

If remediations fail to produce recovery, flap (the same fix re-triggering over
and over), or exceed a rate cap, auto-apply must trip OFF and the engine falls
back to advise-only until reset. This is that mechanism, as an explicit
three-state machine per node key:

    CLOSED  -- normal; auto-apply allowed.
      | (failure_threshold consecutive non-recoveries, OR flap, OR rate cap)
      v
    OPEN    -- tripped; auto-apply refused; advise-only.
      | (after breaker_cooldown_s)
      v
    HALF_OPEN -- allow exactly one trial apply.
      |        success -> CLOSED ;  failure -> OPEN (cooldown restarts)

Time is injected via ``now`` on every call so tests stay synchronous (mirrors
``IdleDetector`` and ``AlertManager`` in the existing codebase).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from et_remediation.config import CapsConfig


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _NodeState:
    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    opened_at: float | None = None
    trip_reason: str = ""
    # timestamps of recent applies (rate cap) and recent (kind, ts) (flap).
    apply_times: deque = field(default_factory=deque)
    kind_times: dict = field(default_factory=dict)  # kind -> deque[ts]


@dataclass(frozen=True)
class BreakerDecision:
    allowed: bool
    state: BreakerState
    reason: str


class CircuitBreaker:
    def __init__(self, caps: CapsConfig | None = None) -> None:
        self.caps = caps or CapsConfig()
        self._nodes: dict[str, _NodeState] = {}

    def _node(self, key: str) -> _NodeState:
        return self._nodes.setdefault(key, _NodeState())

    def state(self, key: str) -> BreakerState:
        return self._node(key).state

    # -- gate ----------------------------------------------------------------

    def allow(self, key: str, now: float) -> BreakerDecision:
        """May an auto-apply proceed on this node right now?"""
        ns = self._node(key)

        if ns.state is BreakerState.OPEN:
            assert ns.opened_at is not None
            if now - ns.opened_at >= self.caps.breaker_cooldown_s:
                ns.state = BreakerState.HALF_OPEN
                return BreakerDecision(True, ns.state, "half_open_trial")
            return BreakerDecision(False, ns.state, ns.trip_reason or "breaker_open")

        if ns.state is BreakerState.HALF_OPEN:
            # A trial is already in flight (recorded via record_apply); refuse
            # further applies until its verify resolves.
            return BreakerDecision(False, ns.state, "half_open_in_flight")

        # CLOSED: enforce the rate cap before allowing.
        self._evict(ns.apply_times, now, self.caps.window_s)
        if len(ns.apply_times) >= self.caps.max_actions_per_window:
            self._trip(ns, now, "rate_capped")
            return BreakerDecision(False, ns.state, "rate_capped")
        return BreakerDecision(True, ns.state, "closed")

    # -- record outcomes -----------------------------------------------------

    def record_apply(self, key: str, kind: str, now: float) -> None:
        """An action was applied; update rate + flap counters, detect flap.

        A HALF_OPEN trial apply is deliberately NOT counted toward rate/flap: it
        is the single probe we allow to test whether the box has recovered, and
        its outcome is recorded via record_success/record_failure. Folding it
        into the deques would let one probe immediately re-trip the breaker.
        """
        ns = self._node(key)
        if ns.state is BreakerState.HALF_OPEN:
            return
        ns.apply_times.append(now)
        self._evict(ns.apply_times, now, self.caps.window_s)

        times = ns.kind_times.setdefault(kind, deque())
        times.append(now)
        self._evict(times, now, self.caps.flap_window_s)
        if len(times) >= self.caps.flap_threshold and ns.state is not BreakerState.OPEN:
            self._trip(ns, now, "flapping")

    def record_success(self, key: str, now: float) -> None:
        """A verify confirmed recovery; clear failures, close the breaker.

        A confirmed recovery means the most recent fix of this kind *worked*, so
        it is not a flap — clear the per-kind flap history so a later, legitimate
        re-fire isn't counted against a fix that already succeeded. The rate
        history (apply_times) is intentionally preserved: the rate cap is about
        actuation frequency regardless of outcome.
        """
        ns = self._node(key)
        ns.consecutive_failures = 0
        ns.kind_times.clear()
        if ns.state in (BreakerState.HALF_OPEN, BreakerState.OPEN):
            ns.state = BreakerState.CLOSED
            ns.opened_at = None
            ns.trip_reason = ""

    def record_failure(self, key: str, now: float) -> None:
        """A verify failed (rolled back); count it, maybe trip."""
        ns = self._node(key)
        ns.consecutive_failures += 1
        if ns.state is BreakerState.HALF_OPEN:
            # The trial failed -> straight back to OPEN, restart cooldown.
            self._trip(ns, now, "half_open_failed")
        elif ns.consecutive_failures >= self.caps.failure_threshold:
            self._trip(ns, now, "repeated_failure")

    # -- admin ---------------------------------------------------------------

    def reset(self, key: str | None = None) -> None:
        """Manually clear a tripped breaker (the documented recovery path)."""
        if key is None:
            self._nodes.clear()
        else:
            self._nodes[key] = _NodeState()

    def _trip(self, ns: _NodeState, now: float, reason: str) -> None:
        ns.state = BreakerState.OPEN
        ns.opened_at = now
        ns.trip_reason = reason

    @staticmethod
    def _evict(dq: deque, now: float, window_s: float) -> None:
        while dq and now - dq[0] > window_s:
            dq.popleft()
