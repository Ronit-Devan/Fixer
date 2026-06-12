"""Bounded per-GPU sample ring buffer.

deque(maxlen=...) gives us O(1) append and amortized O(1) drop-oldest
with hard memory bounds — critical for an always-on daemon.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterator

from gpu_doctor_agent.sampler import Sample


class RingBuffer:
    """Fixed-capacity, time-ordered sample buffer for a single GPU.

    Samples are expected to be appended in monotonic-timestamp order;
    `window()` assumes this and stops at the first sample older than the cutoff.
    """

    __slots__ = ("_capacity", "_buf")

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self._capacity = capacity
        self._buf: Deque[Sample] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def append(self, sample: Sample) -> None:
        self._buf.append(sample)

    def __len__(self) -> int:
        return len(self._buf)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._buf)

    def recent(self, n: int) -> list[Sample]:
        """Last `n` samples in append order. Returns fewer if buffer is shorter."""
        if n <= 0:
            return []
        if n >= len(self._buf):
            return list(self._buf)
        # Slicing a deque requires conversion; for typical n this is cheap
        # compared to a full materialization.
        return list(self._buf)[-n:]

    def window(self, seconds: float, now: float) -> list[Sample]:
        """Samples with `timestamp_s >= now - seconds`.

        Walks backwards from the tail so cost is O(window) not O(buffer).
        """
        if seconds < 0:
            return []
        cutoff = now - seconds
        out: list[Sample] = []
        for s in reversed(self._buf):
            if s.timestamp_s < cutoff:
                break
            out.append(s)
        out.reverse()
        return out

    def mean_util(self, seconds: float, now: float) -> float | None:
        """Mean utilization over the last `seconds`, ignoring samples with util=None.

        Returns None if no valid samples fall in the window.
        """
        win = self.window(seconds, now)
        total = 0.0
        count = 0
        for s in win:
            if s.util_pct is None:
                continue
            total += s.util_pct
            count += 1
        if count == 0:
            return None
        return total / count
