"""Unit tests for the torch.profiler-backed EventSource (NO torch required).

Every test in this file must pass on the no-GPU dev box and in CI where
``torch`` is NOT installed. The whole point of the lazy-import design in
``torch_source.py`` is that the converter + map_category code paths are
pure and feedable with hand-authored dicts; importing the module itself
must never raise.

The GPU-only proofs (live profiler session, real CUDA workload, accuracy
match vs. recorded trace, overhead %) live in ``packages/agent/colab/`` —
NOT here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from gpu_doctor_agent import torch_source as ts_mod
from gpu_doctor_agent.attribution import attribute
from gpu_doctor_agent.detector import IdleEvent
from gpu_doctor_agent.events import EventSource, MockEventSource
from gpu_doctor_agent.torch_source import (
    TorchHookEventSource,
    TorchUnavailable,
    convert_function_events,
    map_category,
)
from gpu_doctor_engine import Event, Verdict


# ---------------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------------


_FIXTURE_PATH: Path = (
    Path(__file__).resolve().parent / "data" / "profiler_events_sync_bound.json"
)


def _load_fixture_events() -> list[dict]:
    with _FIXTURE_PATH.open("rt") as f:
        raw = json.load(f)
    return list(raw["events"])


def _idle_event(started_at_s: float = 0.0) -> IdleEvent:
    return IdleEvent(gpu_index=0, started_at_s=started_at_s, mean_util=0.05)


# ---------------------------------------------------------------------------
# map_category — pure function, exhaustive table-driven
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event,expected",
    [
        # GPU kernel: CUDA device, non-memcpy name.
        ({"name": "volta_sgemm_64x64", "device_type": "DeviceType.CUDA"}, "kernel"),
        # GPU memcpy: CUDA device, memcpy in the name.
        (
            {"name": "Memcpy HtoD (Pageable -> Device)", "device_type": "DeviceType.CUDA"},
            "gpu_memcpy",
        ),
        ({"name": "Memset (Device)", "device_type": "DeviceType.CUDA"}, "gpu_memcpy"),
        # CPU-side CUDA Runtime API call: name starts with "cuda".
        ({"name": "cudaLaunchKernel", "device_type": "DeviceType.CPU"}, "cuda_runtime"),
        (
            {"name": "cudaStreamSynchronize", "device_type": "DeviceType.CPU"},
            "cuda_runtime",
        ),
        (
            {"name": "cudaMemcpyAsync", "device_type": "DeviceType.CPU"},
            "cuda_runtime",
        ),
        # CPU aten op: anything else on CPU side.
        ({"name": "aten::item", "device_type": "DeviceType.CPU"}, "cpu_op"),
        ({"name": "aten::add", "device_type": "DeviceType.CPU"}, "cpu_op"),
        # Lowercase device-type variant (defensive — different torch versions
        # have spelled this enum slightly differently). Substring "cuda" rules.
        ({"name": "some_kernel", "device_type": "cuda"}, "kernel"),
    ],
)
def test_map_category_table(event: dict, expected: str) -> None:
    assert map_category(event) == expected


def test_map_category_handles_missing_fields() -> None:
    """Empty / missing name and device_type degrade to CPU op rather than raise."""
    assert map_category({}) == "cpu_op"
    assert map_category({"name": None, "device_type": None}) == "cpu_op"


# ---------------------------------------------------------------------------
# convert_function_events — pure converter on recorded dicts
# ---------------------------------------------------------------------------


def test_convert_function_events_preserves_timing_and_assigns_category() -> None:
    """Each dict becomes an Event with ts/dur in microseconds and the right cat."""
    dicts = [
        {"name": "volta_sgemm", "device_type": "DeviceType.CUDA", "ts": 100, "dur": 50, "tid": 7},
        {"name": "aten::item", "device_type": "DeviceType.CPU", "ts": 200, "dur": 5, "tid": 1},
        {
            "name": "cudaLaunchKernel",
            "device_type": "DeviceType.CPU",
            "ts": 90,
            "dur": 3,
            "tid": 1,
        },
        {
            "name": "Memcpy HtoD",
            "device_type": "DeviceType.CUDA",
            "ts": 300,
            "dur": 12,
            "tid": 7,
        },
    ]
    events = convert_function_events(dicts)
    assert [e.name for e in events] == [
        "volta_sgemm",
        "aten::item",
        "cudaLaunchKernel",
        "Memcpy HtoD",
    ]
    assert [e.category for e in events] == [
        "kernel",
        "cpu_op",
        "cuda_runtime",
        "gpu_memcpy",
    ]
    assert [e.ts for e in events] == [100, 200, 90, 300]
    assert [e.dur for e in events] == [50, 5, 3, 12]
    assert [e.tid for e in events] == [7, 1, 1, 7]
    # Every output must be an engine Event instance — not a dict / namedtuple.
    for e in events:
        assert isinstance(e, Event)


def test_convert_function_events_uses_thread_field_as_fallback_tid() -> None:
    """torch FunctionEvent exposes the thread id via ``thread``, not ``tid``."""
    dicts = [
        {"name": "k", "device_type": "DeviceType.CUDA", "ts": 0, "dur": 1, "thread": 42}
    ]
    [e] = convert_function_events(dicts)
    assert e.tid == 42


def test_convert_function_events_clamps_negative_duration() -> None:
    """Defensive: a malformed event with end < start clamps dur to 0."""
    dicts = [
        {"name": "k", "device_type": "DeviceType.CUDA", "ts": 100, "dur": -5}
    ]
    [e] = convert_function_events(dicts)
    assert e.dur == 0


def test_convert_function_events_empty_input() -> None:
    assert convert_function_events([]) == []


# ---------------------------------------------------------------------------
# THE bridge proof: converted events -> engine -> SYNC_BOUND
#
# Same shape the mock + file paths use; if the torch path produced a
# different verdict on equivalent data, the abstraction would be a lie.
# ---------------------------------------------------------------------------


def test_recorded_fixture_drives_engine_to_sync_bound() -> None:
    raw_dicts = _load_fixture_events()
    converted = convert_function_events(raw_dicts)

    # Sanity: the fixture should produce the same category split that
    # build_sync_bound_events does — half kernels, half CPU sync ops.
    cats = [e.category for e in converted]
    assert cats.count("kernel") == 5
    assert cats.count("cpu_op") == 5

    # Wrap in a MockEventSource so we exercise the same attribute() path
    # the daemon uses (this is the torch-free way to hand a list of Events
    # to attribute() without standing up the real TorchHookEventSource).
    src = MockEventSource(converted)
    diag = attribute(src, _idle_event(started_at_s=0.0), now_s=5.0)
    assert diag is not None
    assert diag.verdict == Verdict.SYNC_BOUND


# ---------------------------------------------------------------------------
# TorchUnavailable behaviour without torch
#
# These tests force-simulate the no-torch state regardless of whether
# torch happens to be installed on the test runner, by monkeypatching the
# ``_TORCH_AVAILABLE`` flag the module reads inside ``start()``.
# ---------------------------------------------------------------------------


def test_module_imports_without_torch() -> None:
    """Importing torch_source must succeed even with no torch on the path.

    The fact that THIS test file imported ``torch_source`` at module load
    without raising already proves the invariant — we just check the
    public surface is intact. (Deliberately NOT ``importlib.reload``: a
    reload would swap the module-level ``TorchUnavailable`` class for a
    fresh one, breaking ``pytest.raises(TorchUnavailable)`` in sibling
    tests because the locally-imported class would no longer match.)
    """
    assert hasattr(ts_mod, "TorchHookEventSource")
    assert hasattr(ts_mod, "TorchUnavailable")
    assert hasattr(ts_mod, "convert_function_events")
    assert hasattr(ts_mod, "map_category")


def test_start_raises_torch_unavailable_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """start() must raise TorchUnavailable, not bare ImportError."""
    monkeypatch.setattr(ts_mod, "_TORCH_AVAILABLE", False)
    src = TorchHookEventSource()
    with pytest.raises(TorchUnavailable):
        src.start()


def test_capture_before_start_returns_empty() -> None:
    """capture() on a never-started source returns [] (no crash, no fabrication)."""
    src = TorchHookEventSource()
    assert src.capture(gpu_index=0, start_s=0.0, end_s=1.0) == []


def test_capture_after_stop_without_session_returns_empty() -> None:
    """Calling stop() on a never-started source is a no-op; capture stays []."""
    src = TorchHookEventSource()
    src.stop()  # idempotent — must not raise
    assert src.capture(gpu_index=0, start_s=0.0, end_s=1.0) == []


def test_torch_hook_event_source_implements_event_source_interface() -> None:
    """Structural check: TorchHookEventSource is an EventSource."""
    assert issubclass(TorchHookEventSource, EventSource)


def test_is_available_reflects_torch_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ts_mod, "_TORCH_AVAILABLE", False)
    assert TorchHookEventSource.is_available() is False
    monkeypatch.setattr(ts_mod, "_TORCH_AVAILABLE", True)
    assert TorchHookEventSource.is_available() is True


# ---------------------------------------------------------------------------
# Sanity: importing torch_source did not silently leak a real torch import
# (the dev-box guarantee).
# ---------------------------------------------------------------------------


def test_no_torch_import_at_module_load_time() -> None:
    """If torch isn't installed on the test runner, sys.modules must not have it.

    On a runner WITH torch installed this test is meaningless (and skipped);
    the point is the no-torch dev-box / CI case where importing torch_source
    must not have triggered a real torch import as a side-effect.
    """
    if ts_mod._TORCH_AVAILABLE:
        pytest.skip("torch is installed on this runner — invariant trivially holds")
    assert "torch" not in sys.modules
