"""Live CUDA-event capture backend using ``torch.profiler``.

This is the Product-A "live" sibling of ``FileEventSource``: rather than
replay a recorded Chrome trace, it stands up a ``torch.profiler.profile``
session on the live process and converts each emitted ``FunctionEvent`` into
an engine-compatible ``Event`` so the same ``attribute()`` + ``diagnose()``
seam fires identically against live and recorded data.

Design constraints
------------------
1. **Importable without torch.** The dev box and CI run without torch
   installed. ``torch`` is therefore imported lazily inside methods (a
   module-level ``try / except ImportError`` only flips a flag); importing
   ``torch_source`` itself must NEVER raise. Touching the lifecycle on a
   torch-less host raises a clear ``TorchUnavailable`` — the same shape as
   ``NvmlUnavailable`` in ``sampler.py``.

2. **Pure, torch-free converter.** The mapping from a profiler event to an
   engine ``Event`` (and the ``map_category`` discriminator inside it) takes
   a lightweight dict — NOT a ``FunctionEvent``. Tests construct these dicts
   from recorded shapes and exercise the converter without torch.

3. **Low overhead.** Only what the engine consumes is captured. Every
   torch.profiler flag we leave off is annotated below with its overhead
   cost so the decision is auditable, not folklore.

4. **Never crash the daemon.** Profiler teardown is wrapped in try/except;
   ``capture()`` before ``start()`` returns ``[]`` instead of raising.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from gpu_doctor_agent.events import EventSource
from gpu_doctor_engine import Event

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy torch availability flag — set at import time, NEVER raises.
# ---------------------------------------------------------------------------
#
# This module must import on the no-GPU dev box and in CI (no torch wheel).
# We probe for torch here only to set a flag; the actual ``torch.profiler``
# symbol is resolved inside the methods that need it. Tests rely on this:
# `import torch_source` is a no-op even with no torch installed.

try:  # pragma: no cover - trivial import branch
    import torch  # type: ignore[import-not-found]  # noqa: F401

    _TORCH_AVAILABLE: bool = True
except Exception:  # ImportError, or torch present but broken (e.g., missing CUDA libs)
    _TORCH_AVAILABLE = False


class TorchUnavailable(RuntimeError):
    """``torch`` is missing or torch.profiler failed to start.

    Mirrors the ``NvmlUnavailable`` pattern in ``sampler.py``: a clear,
    catchable error so callers can degrade to a mock / file source instead
    of propagating a bare ``ImportError`` up the stack.
    """


# ---------------------------------------------------------------------------
# Pure category mapping. Tests feed dicts; no torch import needed.
# ---------------------------------------------------------------------------
#
# Engine category contract (see packages/engine/src/gpu_doctor_engine/ingest.py):
#   GPU_KERNEL_CATS = {"kernel", "gpu_op"}
#   GPU_MEMCPY_CATS = {"gpu_memcpy", "gpu_memset"}
#   CPU_CATS        = {"cpu_op", "python_function", "user_annotation"}
#   plus "cuda_runtime" used by the sync-attribution scan via name match.
#
# torch.profiler FunctionEvent fields we look at:
#   * ``device_type`` — DeviceType enum; stringified to e.g. "DeviceType.CUDA".
#     We lowercase + substring-match "cuda" so the mapping is robust to
#     enum-repr drift across torch versions.
#   * ``name`` — the op / kernel / runtime-API name. For CPU-side runtime
#     calls this is the CUDA Runtime API name verbatim (cudaLaunchKernel,
#     cudaStreamSynchronize, cudaMemcpyAsync). For GPU-side kernels it's
#     the SASS-resolved kernel name ("volta_sgemm_64x64", etc.). For GPU
#     memcpy it's "Memcpy HtoD (Pageable -> Device)" etc.

_MEMCPY_NAME_TOKENS: tuple[str, ...] = ("memcpy", "memset")


def map_category(event: dict) -> str:
    """Map one profiler event dict to an engine ``Event.category``.

    Pure function. ``event`` MUST contain ``name`` and ``device_type`` keys;
    other fields are ignored. Returns one of the engine categories the
    attribution + diagnose path recognises:

      * ``"kernel"``       — GPU compute kernel (device_type contains "cuda",
                             name does NOT hint memcpy/memset).
      * ``"gpu_memcpy"``   — GPU memcpy/memset activity.
      * ``"cuda_runtime"`` — CPU-side CUDA Runtime API call (e.g.
                             cudaLaunchKernel, cudaStreamSynchronize). The
                             engine's sync-attribution path scans these
                             names regardless of category, but pinning the
                             category preserves the documented contract.
      * ``"cpu_op"``       — any other CPU-side event (aten::*, dataloader,
                             user annotations, etc.).
    """
    name: str = str(event.get("name") or "")
    dev: str = str(event.get("device_type") or "").lower()
    name_lower: str = name.lower()

    if "cuda" in dev:
        # Device side. Memcpy/memset names get their own bucket so the
        # engine's PCIE rules see them; everything else is a kernel.
        if any(tok in name_lower for tok in _MEMCPY_NAME_TOKENS):
            return "gpu_memcpy"
        return "kernel"

    # CPU side. Names starting with "cuda" are Runtime API calls
    # (cudaLaunchKernel, cudaStreamSynchronize, cudaMemcpyAsync, ...).
    if name_lower.startswith("cuda"):
        return "cuda_runtime"
    return "cpu_op"


def _event_dict_to_engine(event: dict) -> Event:
    """Build one engine ``Event`` from a single profiler-event dict.

    Pure function — no torch. ``ts`` and ``dur`` are passed through as the
    engine's native microseconds. ``pid`` defaults to 0 (the profiler does
    not always populate it on FunctionEvent), ``tid`` from the event's
    ``thread`` / ``tid`` field.
    """
    name: str = str(event.get("name") or "")
    ts: int = int(event.get("ts") or 0)
    dur: int = int(event.get("dur") or 0)
    if dur < 0:
        # Defensive: a stale FunctionEvent can have start > end during
        # teardown. Clamp rather than propagate negative durations into
        # the engine's interval math.
        dur = 0
    pid: int = int(event.get("pid") or 0)
    tid: int = int(event.get("tid") or event.get("thread") or 0)
    return Event(
        name=name,
        category=map_category(event),
        pid=pid,
        tid=tid,
        ts=ts,
        dur=dur,
        args={},
    )


def convert_function_events(raw_events: Iterable[dict]) -> list[Event]:
    """Convert an iterable of profiler-event dicts to engine ``Event`` objects.

    Pure, torch-free. The live ``capture()`` path first adapts each
    ``FunctionEvent`` to a dict via ``_function_event_to_portable_dict``,
    then funnels through this converter — so tests can feed hand-authored
    dicts and exercise the exact production conversion code.
    """
    return [_event_dict_to_engine(e) for e in raw_events]


def _function_event_to_portable_dict(fe: Any) -> dict:
    """Adapt a torch ``FunctionEvent`` to the torch-free dict shape.

    The only function in this module that touches torch types — and even
    then via getattr / duck-typing so it survives field renames across
    torch versions. Called inside ``TorchHookEventSource.stop()``.
    """
    # FunctionEvent exposes the time window via ``time_range`` (an
    # ``Interval(start, end)`` in microseconds since the profiler started)
    # on modern torch; older versions used ``cpu_interval`` for CPU events
    # and required computation. We try the modern path first.
    ts: int = 0
    dur: int = 0
    tr = getattr(fe, "time_range", None)
    if tr is not None and hasattr(tr, "start") and hasattr(tr, "end"):
        ts = int(tr.start)
        dur = max(0, int(tr.end) - int(tr.start))
    else:
        # Fallback: older FunctionEvent API
        start = getattr(fe, "start_us", None)
        end = getattr(fe, "end_us", None)
        if start is not None and end is not None:
            ts = int(start() if callable(start) else start)
            end_us = int(end() if callable(end) else end)
            dur = max(0, end_us - ts)

    return {
        "name": str(getattr(fe, "name", "") or ""),
        "ts": ts,
        "dur": dur,
        "tid": int(getattr(fe, "thread", 0) or 0),
        "pid": 0,  # FunctionEvent doesn't carry a pid; engine doesn't need it.
        "device_type": str(getattr(fe, "device_type", "") or ""),
    }


# ---------------------------------------------------------------------------
# Live profiler session — the actual EventSource implementation.
# ---------------------------------------------------------------------------


class TorchHookEventSource(EventSource):
    """``EventSource`` backed by a live ``torch.profiler.profile`` session.

    Lifecycle: ``start()`` opens the profiler, ``stop()`` closes it and
    snapshots the events into an internal cache. ``capture()`` returns the
    cached events (the recorded-trace caveat from ``FileEventSource``
    applies: profiler timestamps are on the profiler's own origin, not the
    agent's monotonic clock, so window-intersection would silently drop
    everything — ``capture()`` therefore ignores its window arguments).

    Overhead notes
    --------------
    Each disabled torch.profiler flag below was a deliberate overhead /
    signal trade. The engine consumes ``name``, ``category``, ``ts``,
    ``dur``, ``tid`` — nothing else — so capturing more is pure cost:

      * ``record_shapes=False``    — collecting input tensor shapes per op
        roughly doubles per-event CPU cost (PyTorch profiler benchmarks)
        and the engine never reads ``args``.
      * ``profile_memory=False``   — adds an allocator-hook callback on
        every ``aten::*`` op (significant overhead in trainers with many
        small tensors) and the engine has no memory rules.
      * ``with_stack=False``       — Python stack capture is the single
        most expensive profiler option (multi-x slowdown observed in
        PyTorch profiler tutorials); we do not surface stacks anywhere.
      * ``with_flops=False``       — FLOPs estimation walks op metadata
        per-event and is unused by the engine.
    """

    # CPU + CUDA activities are the minimum needed to attribute idle to
    # either DataLoader stalls (CPU side) or sync stalls (CPU sync ending
    # just before a GPU idle gap). Adding more activity sets (XPU, MTIA)
    # would be pure overhead on the platforms we target.
    def __init__(self) -> None:
        self._prof: Any = None  # torch.profiler.profile instance when running
        self._events: list[Event] = []
        self._running: bool = False

    @staticmethod
    def is_available() -> bool:
        """Return True iff ``torch`` is importable on this process."""
        return _TORCH_AVAILABLE

    def start(self) -> None:
        """Open a torch.profiler.profile session.

        Raises ``TorchUnavailable`` if torch is not importable on this
        process — caller should fall back to a mock / file source.
        """
        if not _TORCH_AVAILABLE:
            raise TorchUnavailable(
                "torch is not installed; cannot start TorchHookEventSource. "
                "Run on a GPU box or Colab, or use FileEventSource / MockEventSource."
            )
        if self._running:
            return  # idempotent; second start() is a no-op
        try:
            # Imported here (NOT at module top) so the no-torch path stays clean.
            import torch  # type: ignore[import-not-found]
            from torch.profiler import (  # type: ignore[import-not-found]
                ProfilerActivity,
                profile,
            )
        except Exception as e:  # noqa: BLE001 — surface as TorchUnavailable
            raise TorchUnavailable(f"failed to import torch.profiler: {e}") from e

        activities = [ProfilerActivity.CPU]
        # Only attach CUDA activity if a CUDA device is actually visible —
        # otherwise torch.profiler raises on CPU-only hosts.
        try:
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
        except Exception:  # noqa: BLE001 — defensive; treat as no-CUDA
            pass

        try:
            # Construct + start the profiler. Flags chosen for minimum
            # overhead — see class docstring "Overhead notes".
            self._prof = profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_flops=False,
            )
            self._prof.__enter__()
        except Exception as e:  # noqa: BLE001
            self._prof = None
            raise TorchUnavailable(
                f"torch.profiler.profile failed to start: {e}"
            ) from e

        self._events = []
        self._running = True

    def stop(self) -> None:
        """Close the profiler session and snapshot events into the cache.

        Never raises on the daemon path: teardown failures log a warning
        and leave the cache empty. Idempotent — second stop() is a no-op.
        """
        if not self._running:
            return
        self._running = False
        prof = self._prof
        self._prof = None
        if prof is None:
            return
        try:
            prof.__exit__(None, None, None)
        except Exception:
            # Teardown must not propagate — daemon stays alive.
            log.warning(
                "TorchHookEventSource: profiler teardown raised; "
                "events from this session are dropped",
                exc_info=True,
            )
            return
        try:
            # ``events()`` returns the raw per-op FunctionEvent list.
            # Do NOT use ``key_averages()``: that aggregates across calls
            # and loses per-event ``ts`` / ``dur``, which the engine needs
            # for interval math.
            raw = list(prof.events() or [])
        except Exception:
            log.warning(
                "TorchHookEventSource: prof.events() raised; "
                "no events captured this session",
                exc_info=True,
            )
            return
        try:
            dicts = [_function_event_to_portable_dict(fe) for fe in raw]
            self._events = convert_function_events(dicts)
        except Exception:
            log.warning(
                "TorchHookEventSource: event conversion failed; "
                "no events captured this session",
                exc_info=True,
            )
            self._events = []

    def capture(
        self, gpu_index: int, start_s: float, end_s: float
    ) -> list[Event]:
        """Return events captured by the most-recent profiler session.

        Like ``FileEventSource``, the window arguments are ignored: profiler
        timestamps live on the profiler's own origin (microseconds since
        the session started), which is unrelated to the agent's monotonic
        wall clock. Naively intersecting would always yield ``[]`` and
        starve attribution. ``gpu_index`` is also unused — the underlying
        profiler is process-wide, not per-GPU partitioned.

        Returns ``[]`` if no session has ever been completed.
        """
        del gpu_index, start_s, end_s  # see docstring
        return list(self._events)

    # Context-manager sugar so callers can ``with TorchHookEventSource() as src:``
    # in tests and the Colab notebook. Errors during ``__exit__`` are swallowed
    # by ``stop()`` to honour the daemon-never-crashes invariant.
    def __enter__(self) -> "TorchHookEventSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


__all__ = [
    "TorchHookEventSource",
    "TorchUnavailable",
    "convert_function_events",
    "map_category",
]
