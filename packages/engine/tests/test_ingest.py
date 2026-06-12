"""Tests for the gpu_doctor_engine.ingest module.

The unit tests (test_skips_*, test_merges_*) exercise the parsing and
interval-merging logic directly.  The integration test (test_real_trace_loads)
loads a pre-built JSON fixture to validate end-to-end ingestion against a
realistic trace shape.
"""

from __future__ import annotations

import json
from pathlib import Path

from gpu_doctor_engine.ingest import GPU_KERNEL_CATS, _busy_time_us, load_trace
from gpu_doctor_engine.types import Event


# ---------------------------------------------------------------------------
# Unit tests — no real files needed for the core parsing rules
# ---------------------------------------------------------------------------


def test_skips_non_complete_events(tmp_path: Path) -> None:
    """Events whose 'ph' field is not 'X' must be excluded from the Trace.

    The Chrome Trace Event format uses ph='X' for complete (duration) events.
    Other phases like 'B'/'E' (begin/end) and 'M' (metadata) carry no duration
    and should be silently dropped rather than causing a crash.
    """
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(
        json.dumps(
            {
                "traceEvents": [
                    # Only this one should survive
                    {
                        "ph": "X",
                        "name": "kernel_op",
                        "cat": "kernel",
                        "pid": 1,
                        "tid": 1,
                        "ts": 0,
                        "dur": 100,
                    },
                    # Begin event — no dur field, not a complete event
                    {
                        "ph": "B",
                        "name": "begin_marker",
                        "cat": "kernel",
                        "pid": 1,
                        "tid": 1,
                        "ts": 0,
                    },
                    # Metadata event — should be ignored
                    {"ph": "M", "name": "thread_name", "pid": 1, "tid": 1},
                ]
            }
        )
    )

    trace = load_trace(trace_file)

    assert len(trace.events) == 1
    assert trace.events[0].name == "kernel_op"


def test_skips_metadata_with_string_tid(tmp_path: Path) -> None:
    """Events with a non-numeric 'tid' (e.g. 'Spans') are skipped without crashing.

    Some PyTorch profiler traces emit span/metadata rows with tid set to a
    descriptive string.  _safe_int() returns None for those, which triggers the
    None-guard that drops the event before it can pollute the Event list.
    """
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(
        json.dumps(
            {
                "traceEvents": [
                    # Valid event — numeric tid
                    {
                        "ph": "X",
                        "name": "real_kernel",
                        "cat": "kernel",
                        "pid": 1,
                        "tid": 1,
                        "ts": 0,
                        "dur": 100,
                    },
                    # String tid — _safe_int returns None → must be dropped
                    {
                        "ph": "X",
                        "name": "span_event",
                        "cat": "fwd",
                        "pid": 1,
                        "tid": "Spans",
                        "ts": 0,
                        "dur": 50,
                    },
                ]
            }
        )
    )

    trace = load_trace(trace_file)

    assert len(trace.events) == 1
    assert trace.events[0].name == "real_kernel"


def test_merges_overlapping_kernel_intervals() -> None:
    """Overlapping kernel events on different streams count as their union, not their sum.

    kernel_a runs [0, 100], kernel_b runs [50, 150].  Their union is [0, 150]
    = 150µs.  Summing durations naively would give 200µs — a 33% overcount
    that would inflate gpu_utilization.
    """
    events = [
        Event(name="kernel_a", category="kernel", pid=1, tid=1, ts=0, dur=100),
        # Overlaps with kernel_a; together they span [0, 150]
        Event(name="kernel_b", category="kernel", pid=1, tid=2, ts=50, dur=100),
    ]

    busy = _busy_time_us(events, GPU_KERNEL_CATS)

    assert busy == 150


# ---------------------------------------------------------------------------
# Integration test — real fixture file
# ---------------------------------------------------------------------------


def test_real_trace_loads() -> None:
    """Loading the dataloader_starved fixture yields thousands of events and
    a non-zero GPU busy time.

    This validates end-to-end ingestion: JSON parsing, event filtering, interval
    merging, and Trace field population all run against a realistic-shape trace.
    """
    fixture = Path(__file__).parent / "fixtures" / "dataloader_starved.json"
    trace = load_trace(fixture)

    assert (
        len(trace.events) >= 2000
    ), f"Expected ≥2000 events in fixture, got {len(trace.events)}"
    assert trace.gpu_kernel_time_us > 0, "Expected non-zero GPU busy time in fixture"
