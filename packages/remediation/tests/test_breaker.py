"""Circuit breaker: trips on failure / flap / rate; half-open recovery."""

from __future__ import annotations

from et_remediation.breaker import BreakerState, CircuitBreaker
from et_remediation.config import CapsConfig

K = "node-0"


def test_repeated_failure_trips():
    cb = CircuitBreaker(CapsConfig(failure_threshold=3))
    assert cb.allow(K, 0).allowed
    cb.record_failure(K, 1)
    cb.record_failure(K, 2)
    assert cb.state(K) is BreakerState.CLOSED
    cb.record_failure(K, 3)  # third
    assert cb.state(K) is BreakerState.OPEN
    assert not cb.allow(K, 4).allowed


def test_flap_trips():
    cb = CircuitBreaker(CapsConfig(flap_threshold=3, max_actions_per_window=100))
    cb.record_apply(K, "set_power_limit", 0)
    cb.record_apply(K, "set_power_limit", 1)
    assert cb.state(K) is BreakerState.CLOSED
    cb.record_apply(K, "set_power_limit", 2)  # same kind, third time
    assert cb.state(K) is BreakerState.OPEN
    assert cb.allow(K, 3).reason == "flapping"


def test_rate_cap_trips():
    # Distinct kinds so this isolates the RATE cap (3 total), not the flap rule.
    cb = CircuitBreaker(CapsConfig(max_actions_per_window=3, window_s=600, flap_threshold=10))
    for i in range(3):
        assert cb.allow(K, i).allowed
        cb.record_apply(K, f"kind{i}", i)
    d = cb.allow(K, 3)  # 4th within window
    assert not d.allowed and d.reason == "rate_capped"
    assert cb.state(K) is BreakerState.OPEN


def test_rate_window_evicts_old_applies():
    cb = CircuitBreaker(CapsConfig(max_actions_per_window=2, window_s=10))
    cb.record_apply(K, "k", 0)
    cb.record_apply(K, "k", 1)
    # 100s later the old applies have aged out of the window.
    assert cb.allow(K, 100).allowed


def test_half_open_then_success_closes():
    cb = CircuitBreaker(CapsConfig(failure_threshold=1, breaker_cooldown_s=50))
    cb.record_failure(K, 0)  # trips
    assert cb.state(K) is BreakerState.OPEN
    assert not cb.allow(K, 10).allowed  # before cooldown
    d = cb.allow(K, 60)  # after cooldown -> half-open trial
    assert d.allowed and d.state is BreakerState.HALF_OPEN
    cb.record_success(K, 61)
    assert cb.state(K) is BreakerState.CLOSED


def test_half_open_then_failure_reopens():
    cb = CircuitBreaker(CapsConfig(failure_threshold=1, breaker_cooldown_s=50))
    cb.record_failure(K, 0)
    cb.allow(K, 60)  # -> half-open
    cb.record_failure(K, 61)  # trial fails
    assert cb.state(K) is BreakerState.OPEN
    assert not cb.allow(K, 70).allowed  # cooldown restarted


def test_manual_reset_clears():
    cb = CircuitBreaker(CapsConfig(failure_threshold=1))
    cb.record_failure(K, 0)
    assert cb.state(K) is BreakerState.OPEN
    cb.reset(K)
    assert cb.state(K) is BreakerState.CLOSED
    assert cb.allow(K, 1).allowed
