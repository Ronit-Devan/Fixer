"""Simulation harness: drive the manager through ticks with no GPU.

A ``Sim`` wires a ``RemediationManager`` to a ``FakeActuator`` + ``FakeTelemetryModel``
and exposes a single ``tick(diagnosis)`` that advances a virtual clock, hands the
manager the current telemetry window, and returns the ``Outcome``. This is the
remediation analogue of ``et_monitor.demo`` — a deterministic timeline driver.
"""

from __future__ import annotations

from et_remediation import (
    CircuitBreaker,
    FakeActuator,
    FakeTelemetryModel,
    ProtectedWorkload,
    RemediationConfig,
    RemediationManager,
    RemediationMode,
    default_registry,
)


class Sim:
    def __init__(
        self,
        *,
        mode: RemediationMode = RemediationMode.AUTO,
        model: FakeTelemetryModel | None = None,
        recover_on_apply: bool = True,
        fail_kinds=None,
        verify_window_s: float = 5.0,
        protected_pids: list[int] | None = None,
        config: RemediationConfig | None = None,
        window_n: int = 3,
        node_id: str = "node-0",
    ) -> None:
        self.model = model or FakeTelemetryModel()
        self.actuator = FakeActuator(
            self.model, recover_on_apply=recover_on_apply, fail_kinds=fail_kinds
        )
        self.cfg = config or RemediationConfig(mode=mode, verify_window_s=verify_window_s)
        if protected_pids is not None:
            self.cfg.protected_pids = protected_pids
        self.window_n = window_n
        self._t = 0.0
        self.breaker = CircuitBreaker(self.cfg.caps)
        self.mgr = RemediationManager(
            default_registry(),
            self.cfg,
            [self.actuator],
            breaker=self.breaker,
            protected=ProtectedWorkload(
                pids=frozenset(self.cfg.protected_pids), label=self.cfg.protected_label
            ),
            now_fn=lambda: self._t,
            node_id=node_id,
        )

    def tick(self, diagnosis, *, source: str = "monitor", dt: float = 1.0):
        out = self.mgr.observe(diagnosis, self.model.window(self.window_n), now=self._t, source=source)
        self._t += dt
        return out

    def run(self, diagnosis, n: int, *, source: str = "monitor"):
        return [self.tick(diagnosis, source=source).kind.value for _ in range(n)]
