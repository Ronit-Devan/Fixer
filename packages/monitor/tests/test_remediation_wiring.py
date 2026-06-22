"""The monitor <-> remediation seam, driven through the real Monitor.tick().

Proves the layer plugs in exactly parallel to alert_manager: the live diagnosis
+ window reach the manager every tick, a non-disruptive fix auto-applies and then
confirms once the (mocked) telemetry recovers, a failing manager never takes the
sampling loop down, and the HTTP endpoints expose status / audit / kill-switch.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from et_monitor.gpu import MockGpuSampler
from et_monitor.server import create_app
from et_monitor.state import Monitor, MonitorConfig

from et_remediation import (
    ActionKind,
    FakeActuator,
    FakeTelemetryModel,
    RemediationConfig,
    RemediationManager,
    RemediationMode,
    default_registry,
)
from et_remediation.actuators.base import ActuationState, Actuator


# An actuator that "fixes" the real monitor's mock sampler: raising the SM clock
# is what relieves a thermal throttle, so the next diagnosis recovers.
class _ClockFixActuator(Actuator):
    backend = "clockfix"

    def __init__(self, sampler: MockGpuSampler) -> None:
        self.sampler = sampler
        self.applied = 0

    def capabilities(self):
        return {ActionKind.SET_POWER_LIMIT}

    def snapshot_state(self, req):
        return ActuationState(req.kind, req.target, {"sm_clock_mhz": self.sampler.sm_clock_mhz})

    def apply(self, req):
        from et_remediation import ActionResult

        self.applied += 1
        if not req.dry_run:
            self.sampler.sm_clock_mhz = int(self.sampler._sm_max * 0.98)  # recovered
        return ActionResult(ok=True, kind=req.kind, command="clockfix", executed=not req.dry_run)

    def rollback(self, req, prior):
        from et_remediation import ActionResult

        return ActionResult(ok=True, kind=req.kind, command="clockfix-rollback", executed=True)


def _throttled_sampler() -> MockGpuSampler:
    s = MockGpuSampler()
    s.util_pct = 85.0            # under load
    s.sm_clock_mhz = int(s._sm_max * 0.45)  # clock dragged down -> throttle
    s.temp_c = 84.0
    return s


def _manager(sampler, actuators, mode=RemediationMode.AUTO, vw=2.0):
    cfg = RemediationConfig(mode=mode, verify_window_s=vw)
    return RemediationManager(default_registry(), cfg, actuators, now_fn=lambda: 0.0)


def test_tick_drives_remediation_and_auto_applies():
    sampler = _throttled_sampler()
    fixer = _ClockFixActuator(sampler)
    mon = Monitor(sampler, None, MonitorConfig(interval_s=1.0))
    mon.remediation_manager = _manager(sampler, [fixer])
    for _ in range(4):
        mon.tick()
    # The fix was applied through the live loop and audited.
    assert fixer.applied >= 1
    phases = [r.phase.value for r in mon.remediation_manager.audit.recent()]
    assert "apply" in phases


def test_tick_confirms_recovery_after_fix():
    sampler = _throttled_sampler()
    mon = Monitor(sampler, None, MonitorConfig(interval_s=1.0))
    # now_fn advances with ticks so the verify window can elapse / resolve.
    clock = {"t": 0.0}
    cfg = RemediationConfig(mode=RemediationMode.AUTO, verify_window_s=5.0)
    mon.remediation_manager = RemediationManager(
        default_registry(), cfg, [_ClockFixActuator(sampler)], now_fn=lambda: clock["t"]
    )
    seen = []
    for _ in range(8):
        clock["t"] += 1.0
        mon.tick()
        seen.append(mon.remediation_manager.status()["state"])
    phases = [r.phase.value for r in mon.remediation_manager.audit.recent()]
    # Applied, then confirmed once the clock recovered in a later window.
    assert "apply" in phases and "verify" in phases


def test_tick_never_dies_if_remediation_raises():
    class Boom:
        def observe(self, *a, **k):
            raise RuntimeError("kaboom")

    mon = Monitor(MockGpuSampler(), None, MonitorConfig())
    mon.remediation_manager = Boom()
    snap = mon.tick()  # must not raise
    assert snap is not None


def _client_with_remediation():
    sampler = _throttled_sampler()
    mon = Monitor(sampler, None, MonitorConfig(interval_s=1.0))
    mon.remediation_manager = _manager(sampler, [FakeActuator(FakeTelemetryModel())])
    for _ in range(4):
        mon.tick()
    return TestClient(create_app(mon, start=False)), mon


def test_endpoint_state_and_audit():
    client, _ = _client_with_remediation()
    state = client.get("/api/remediation/state").json()
    assert state["enabled"] is True
    assert state["mode"] == "auto"
    audit = client.get("/api/remediation/audit").json()
    assert "records" in audit


def test_endpoint_mode_kill_switch():
    client, mon = _client_with_remediation()
    r = client.post("/api/remediation/mode", json={"mode": "advise"})
    assert r.json()["mode"] == "advise"
    assert mon.remediation_manager.mode is RemediationMode.ADVISE
    # invalid mode rejected
    assert client.post("/api/remediation/mode", json={"mode": "nope"}).status_code == 400


def test_state_endpoint_reports_disabled_without_manager():
    mon = Monitor(MockGpuSampler(), None, MonitorConfig())
    client = TestClient(create_app(mon, start=False))
    assert client.get("/api/remediation/state").json() == {"enabled": False}
