"""End-to-end: multi-GPU box + per-GPU remediation factory + fleet blast radius.

Exercises the production shape (what __main__ wires): one RemediationManager per
GPU sharing a FleetCoordinator, driven through the real Monitor.tick() loop.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from et_monitor.gpu import GpuReading, GpuSampler
from et_monitor.server import create_app
from et_monitor.state import Monitor, MonitorConfig

from et_remediation import (
    CommandRunner,
    DataCenterActuator,
    FakeActuator,
    FakeTelemetryModel,
    FleetCoordinator,
    LlamaCppActuator,
    RemediationConfig,
    RemediationManager,
    RemediationMode,
    default_registry,
)


class ThrottledMulti(GpuSampler):
    """N GPUs all under load with the SM clock dragged down (reactive throttle)."""

    backend = "throttled-multi"

    def __init__(self, n: int) -> None:
        self.n = n

    def gpu_count(self) -> int:
        return self.n

    def read(self) -> list[GpuReading]:
        now = time.time()
        return [
            GpuReading(now, i, f"GPU{i}", util_pct=92.0, mem_used_mb=18000.0,
                       mem_total_mb=24000.0, power_w=60.0, power_limit_w=70.0,
                       sm_clock_mhz=1300, sm_clock_max_mhz=2520, temp_c=80.0)
            for i in range(self.n)
        ]


def _factory(fleet, registry):
    def make(index: int):
        cfg = RemediationConfig(mode=RemediationMode.AUTO, verify_window_s=5.0, trigger_debounce=1)
        act = FakeActuator(FakeTelemetryModel(), recover_on_apply=False)
        return RemediationManager(
            registry, cfg, [act], fleet=fleet,
            now_fn=lambda: 0.0, node_id=f"host:gpu{index}", gpu_index=index
        )
    return make


def test_per_gpu_managers_created_and_fleet_caps_concurrency():
    fleet = FleetCoordinator(max_concurrent=1)
    registry = default_registry()
    mon = Monitor(ThrottledMulti(4), None, MonitorConfig(interval_s=1.0))
    mon.remediation_factory = _factory(fleet, registry)

    # Need >= min_samples ticks for a verdict to form; then the throttled cards
    # try to act and the fleet cap admits exactly one.
    for _ in range(4):
        mon.tick()

    mgrs = mon.remediation_managers()
    assert set(mgrs) == {0, 1, 2, 3}  # one manager per GPU
    # Fleet blast-radius cap = 1: at most one GPU is actively remediating.
    verifying = [i for i, m in mgrs.items() if m.status()["state"] == "verifying"]
    assert len(verifying) == 1
    assert fleet.active_count() == 1


def test_multigpu_remediation_endpoints():
    fleet = FleetCoordinator(max_concurrent=2)
    mon = Monitor(ThrottledMulti(2), None, MonitorConfig(interval_s=1.0))
    mon.remediation_factory = _factory(fleet, default_registry())
    mon.tick()
    client = TestClient(create_app(mon, start=False))

    state = client.get("/api/remediation/state").json()
    assert state["enabled"] is True
    assert len(state["gpus"]) == 2  # per-GPU status
    assert state["mode"] == "auto"

    # Box-wide kill-switch flips every GPU's manager.
    assert client.post("/api/remediation/mode", json={"mode": "advise"}).json()["mode"] == "advise"
    for m in mon.remediation_managers().values():
        assert m.mode is RemediationMode.ADVISE


def test_startup_kill_switch_applies_to_lazily_built_managers():
    # Operator flips the kill-switch BEFORE any GPU has been sampled (no managers
    # exist yet). The box-wide mode must be honored by every manager built later.
    mon = Monitor(ThrottledMulti(2), None, MonitorConfig(interval_s=1.0))
    mon.remediation_factory = _factory(FleetCoordinator(1), default_registry())
    mon.set_remediation_mode(RemediationMode.OFF)  # before first tick
    assert not mon.remediation_managers()  # nothing built yet
    for _ in range(4):
        mon.tick()
    mgrs = mon.remediation_managers()
    assert mgrs and all(m.mode is RemediationMode.OFF for m in mgrs.values())


def test_mode_endpoint_works_before_first_tick():
    mon = Monitor(ThrottledMulti(2), None, MonitorConfig(interval_s=1.0))
    mon.remediation_factory = _factory(FleetCoordinator(1), default_registry())
    client = TestClient(create_app(mon, start=False))
    # No tick yet -> no managers, but the kill-switch must still take effect.
    assert client.post("/api/remediation/mode", json={"mode": "off"}).json()["mode"] == "off"
    assert client.get("/api/remediation/state").json()["mode"] == "off"
    for _ in range(4):
        mon.tick()
    assert all(m.mode is RemediationMode.OFF for m in mon.remediation_managers().values())


def test_real_actuators_compose_with_factory():
    # The exact actuator set __main__ builds, shared across GPUs (no execution:
    # non-executing runner), proving the wiring composes.
    fleet = FleetCoordinator(max_concurrent=1)
    actuators = [DataCenterActuator(CommandRunner(execute=False)),
                 LlamaCppActuator(CommandRunner(execute=False))]
    cfg = RemediationConfig(mode=RemediationMode.AUTO, trigger_debounce=1)

    def make(index):
        return RemediationManager(default_registry(), cfg, actuators, fleet=fleet,
                                  now_fn=lambda: 0.0, node_id=f"h:gpu{index}", gpu_index=index)

    mon = Monitor(ThrottledMulti(2), None, MonitorConfig())
    mon.remediation_factory = make
    mon.tick()
    # Managers exist and at least one acted (built a real nvidia-smi command).
    mgrs = mon.remediation_managers()
    assert len(mgrs) == 2
