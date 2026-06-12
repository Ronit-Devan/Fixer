"""IdleDetector FSM: sustain gating, edge-triggered events, hysteresis."""

from __future__ import annotations

import pytest

from gpu_doctor_agent.config import AgentConfig
from gpu_doctor_agent.detector import IdleDetector, IdleEvent, IdleState


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


def _config(**overrides: object) -> AgentConfig:
    defaults: dict[str, object] = dict(
        sample_interval_s=1.0,
        idle_util_threshold=0.20,
        idle_sustain_s=5.0,
        recovery_util_threshold=0.40,
        ring_capacity=600,
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)  # type: ignore[arg-type]


def test_starts_healthy() -> None:
    d = IdleDetector(gpu_index=0, config=_config(), now_fn=_FakeClock())
    assert d.state is IdleState.HEALTHY


def test_no_event_before_sustain_elapses() -> None:
    clock = _FakeClock()
    d = IdleDetector(gpu_index=0, config=_config(), now_fn=clock)
    # Drop into idle at t=0.
    assert d.observe(0.10) is None
    assert d.state is IdleState.IDLE_SUSPECTED
    # 4s elapses (< sustain=5s). Still no event.
    clock.advance(4.0)
    assert d.observe(0.10) is None
    assert d.state is IdleState.IDLE_SUSPECTED


def test_exactly_one_event_on_confirmation() -> None:
    clock = _FakeClock()
    d = IdleDetector(gpu_index=2, config=_config(), now_fn=clock)
    d.observe(0.10)  # t=0: suspect
    clock.advance(5.0)
    ev = d.observe(0.10)  # t=5: confirm
    assert isinstance(ev, IdleEvent)
    assert ev.gpu_index == 2
    assert ev.started_at_s == 0.0
    assert d.state is IdleState.IDLE_CONFIRMED
    # Further idle samples must NOT re-fire the event.
    for _ in range(10):
        clock.advance(1.0)
        assert d.observe(0.10) is None
    assert d.state is IdleState.IDLE_CONFIRMED


def test_brief_dip_recovers_without_event() -> None:
    clock = _FakeClock()
    d = IdleDetector(gpu_index=0, config=_config(), now_fn=clock)
    d.observe(0.10)  # suspect
    assert d.state is IdleState.IDLE_SUSPECTED
    clock.advance(2.0)
    # Util pops back up above the entry threshold before sustain elapses.
    assert d.observe(0.30) is None
    assert d.state is IdleState.HEALTHY


def test_hysteresis_recovery_requires_higher_threshold() -> None:
    clock = _FakeClock()
    d = IdleDetector(gpu_index=0, config=_config(), now_fn=clock)
    d.observe(0.10)
    clock.advance(5.0)
    ev = d.observe(0.10)
    assert ev is not None
    assert d.state is IdleState.IDLE_CONFIRMED

    # Crossing the *entry* threshold (0.20) is not enough to recover.
    clock.advance(1.0)
    assert d.observe(0.25) is None
    assert d.state is IdleState.IDLE_CONFIRMED
    assert d.observe(0.39) is None
    assert d.state is IdleState.IDLE_CONFIRMED

    # Crossing the recovery threshold (0.40) snaps back to HEALTHY.
    assert d.observe(0.40) is None
    assert d.state is IdleState.HEALTHY


def test_recovery_then_re_idle_emits_new_event() -> None:
    clock = _FakeClock()
    d = IdleDetector(gpu_index=0, config=_config(), now_fn=clock)
    d.observe(0.10)
    clock.advance(5.0)
    first = d.observe(0.10)
    assert first is not None

    # Recover above 0.40.
    clock.advance(1.0)
    d.observe(0.80)
    assert d.state is IdleState.HEALTHY

    # New idle episode: must require sustain again, then emit a fresh event.
    clock.advance(1.0)
    d.observe(0.10)
    assert d.state is IdleState.IDLE_SUSPECTED
    clock.advance(5.0)
    second = d.observe(0.10)
    assert second is not None
    assert second is not first
    assert second.started_at_s > first.started_at_s


def test_none_util_holds_state() -> None:
    """A sampler-read failure must not promote or recover state."""
    clock = _FakeClock()
    d = IdleDetector(gpu_index=0, config=_config(), now_fn=clock)
    d.observe(0.10)  # suspect
    clock.advance(5.0)
    # NVML read failed. State must NOT advance to CONFIRMED on no information.
    assert d.observe(None) is None
    assert d.state is IdleState.IDLE_SUSPECTED
    # A subsequent real reading still idle -> confirm now.
    ev = d.observe(0.10)
    assert ev is not None


def test_sustain_zero_confirms_immediately() -> None:
    clock = _FakeClock()
    d = IdleDetector(gpu_index=0, config=_config(idle_sustain_s=0.0), now_fn=clock)
    ev = d.observe(0.10)
    assert ev is not None
    assert d.state is IdleState.IDLE_CONFIRMED


def test_config_rejects_recovery_below_entry() -> None:
    with pytest.raises(Exception):
        # recovery must be strictly greater than entry threshold (hysteresis)
        _config(idle_util_threshold=0.40, recovery_util_threshold=0.30)
