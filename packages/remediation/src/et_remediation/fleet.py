"""Fleet-wide blast-radius coordinator.

On a multi-GPU box (or a fleet sharing one coordinator) the per-GPU managers are
otherwise independent — each would happily auto-apply at the same instant. That
is exactly the blast radius we must bound: a bad-but-plausible fix shouldn't fire
on every card simultaneously. This coordinator caps how many keys (GPUs / nodes)
may be *actively remediating* (holding an in-flight verify) at once. A manager
acquires a slot before it actuates and releases it when the verify resolves; if
no slot is free it falls back to advise-only.

Thread-safe: managers run on independent sampling threads.
"""

from __future__ import annotations

import threading


class FleetCoordinator:
    def __init__(self, max_concurrent: int = 1) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.max_concurrent = max_concurrent
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def try_acquire(self, key: str) -> bool:
        """Reserve a remediation slot for ``key``. True if granted (or already held)."""
        with self._lock:
            if key in self._active:
                return True  # re-entrant: this key already holds its slot
            if len(self._active) >= self.max_concurrent:
                return False
            self._active.add(key)
            return True

    def release(self, key: str) -> None:
        with self._lock:
            self._active.discard(key)

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def active(self) -> set[str]:
        with self._lock:
            return set(self._active)
