"""Trend math: slope and time-to-threshold projection (predictive primitives)."""

from __future__ import annotations

from et_monitor.trends import r_squared, slope_per_s, time_to_threshold


def test_r_squared_perfect_line_and_noise():
    assert abs(r_squared([0, 1, 2, 3], [0, 2, 4, 6]) - 1.0) < 1e-9  # perfect fit
    # Flat values have no trend to be confident about.
    assert r_squared([0, 1, 2], [5, 5, 5]) is None
    # A zig-zag has a low r-squared.
    r = r_squared([0, 1, 2, 3, 4, 5], [70, 83, 61, 84, 62, 83])
    assert r is not None and r < 0.6


def test_slope_basic():
    assert slope_per_s([0, 1, 2, 3], [0, 2, 4, 6]) == 2.0
    assert slope_per_s([0, 1, 2], [5, 5, 5]) == 0.0


def test_slope_none_cases():
    assert slope_per_s([0], [1]) is None  # one point
    assert slope_per_s([2, 2, 2], [1, 2, 3]) is None  # zero time variance
    assert slope_per_s([], []) is None


def test_time_to_threshold_rising():
    # value 80 -> 92 over 4s (slope 3/s); to 95 is 1s from the last point (92).
    t = time_to_threshold([0, 1, 2, 3, 4], [80, 83, 86, 89, 92], 95, rising=True)
    assert abs(t - 1.0) < 1e-6


def test_time_to_threshold_falling_clock():
    # clock 0.95 -> 0.75 over 4s; to a 0.70 floor is 1s more.
    t = time_to_threshold([0, 1, 2, 3, 4], [0.95, 0.90, 0.85, 0.80, 0.75], 0.70, rising=False)
    assert abs(t - 1.0) < 1e-6


def test_time_to_threshold_not_approaching_returns_none():
    # Flat -> never crosses.
    assert time_to_threshold([0, 1, 2], [50, 50, 50], 95) is None
    # Already past the target.
    assert time_to_threshold([0, 1, 2], [96, 97, 98], 95, rising=True) is None
    # Moving the wrong way (falling) when we asked about a rising crossing.
    assert time_to_threshold([0, 1, 2], [90, 80, 70], 95, rising=True) is None
