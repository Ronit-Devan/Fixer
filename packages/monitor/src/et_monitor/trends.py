"""Trend math for *predictive* detection — catching a problem before it lands.

The reactive verdicts fire on the current aggregate state (clock already low, KV
already full). To stop a problem while it is still forming, we fit a line to a
metric over the recent window and project when it will cross a danger threshold.
If that crossing is close enough, we raise an early ("predicted") verdict so the
remediation layer can act with lead time.

Pure functions over (timestamps, values). No I/O, no clocks — fully testable.
"""

from __future__ import annotations

from statistics import mean
from typing import Sequence


def slope_per_s(times: Sequence[float], values: Sequence[float]) -> float | None:
    """Least-squares slope d(value)/d(time). None if not derivable.

    Returns None for fewer than 2 points or a degenerate (zero-variance) time
    axis, so callers never divide by zero or trust a one-point 'trend'.
    """
    n = len(times)
    if n < 2 or len(values) != n:
        return None
    mt = mean(times)
    mv = mean(values)
    den = sum((t - mt) ** 2 for t in times)
    if den == 0:
        return None
    num = sum((t - mt) * (v - mv) for t, v in zip(times, values))
    return num / den


def r_squared(times: Sequence[float], values: Sequence[float]) -> float | None:
    """Goodness of fit of the linear trend in [0, 1]. None if not derivable.

    Used to reject *noise*: a window where the metric jitters with no real trend
    has a low r², so we don't raise a confident early warning off a line fit
    through scatter. Flat values (zero variance) return None — there is no trend
    to be confident about.
    """
    n = len(times)
    if n < 2 or len(values) != n:
        return None
    mt = mean(times)
    mv = mean(values)
    st = sum((t - mt) ** 2 for t in times)
    sv = sum((v - mv) ** 2 for v in values)
    if st == 0 or sv == 0:
        return None
    cov = sum((t - mt) * (v - mv) for t, v in zip(times, values))
    r = cov / ((st**0.5) * (sv**0.5))
    return r * r


def time_to_threshold(
    times: Sequence[float],
    values: Sequence[float],
    target: float,
    *,
    rising: bool = True,
) -> float | None:
    """Seconds from the latest sample until the linear fit crosses ``target``.

    ``rising=True`` projects an increasing metric (temperature, KV, VRAM) up to a
    ceiling; ``rising=False`` projects a decreasing metric (SM clock) down to a
    floor. Returns None when the metric is not moving toward the threshold, is
    already past it, or the trend is too flat to project — i.e. "no imminent
    crossing", which the caller treats as "don't predict".
    """
    s = slope_per_s(times, values)
    if s is None or s == 0:
        return None
    last_v = values[-1]
    if rising:
        if s <= 0 or last_v >= target:
            return None
    else:
        if s >= 0 or last_v <= target:
            return None
    dt = (target - last_v) / s
    return dt if dt > 0 else None
