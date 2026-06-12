"""End-to-end consistency: FileEventSource attribution == engine diagnose().

The bridge from a real recorded PyTorch Profiler trace through the agent's
``FileEventSource`` -> ``attribute()`` path MUST produce the same ``Verdict``
the engine itself produces from ``diagnose(load_trace(path))``. Anything
less means the agent silently disagrees with the engine on the same input,
which is the most insidious class of attribution bug.

The supporting tests also pin two invariants the consistency guarantee
relies on:

  * A bad / missing trace path degrades to ``None`` — the daemon never
    crashes on attribution.
  * The ``Trace`` the agent hands to ``diagnose()`` has its ``duration_us``
    derived from the captured events' own time span, not from the live
    wall-clock window (which is meaningless for a recorded trace whose
    timestamps live on the profiler's own origin).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_doctor_agent import attribution as attribution_mod
from gpu_doctor_agent.attribution import attribute
from gpu_doctor_agent.detector import IdleEvent
from gpu_doctor_agent.events import FileEventSource
from gpu_doctor_engine import Trace, Verdict, diagnose, load_trace


# ---------------------------------------------------------------------------
# Fixture discovery: walk up from this test file to find the repo's
# top-level fixtures/ directory. Robust to being invoked from any cwd
# (pytest from packages/agent, from the repo root, from a CI runner, etc.).
# ---------------------------------------------------------------------------


def _find_fixtures_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        candidate = parent / "fixtures"
        if candidate.is_dir() and (candidate / "dataloader_starved.json").is_file():
            return candidate
    raise FileNotFoundError(
        "could not locate top-level fixtures/ directory walking up from "
        f"{here}"
    )


_FIXTURES_DIR: Path = _find_fixtures_dir()


def _idle_event(started_at_s: float = 0.0) -> IdleEvent:
    """A throwaway IdleEvent — FileEventSource ignores the window entirely."""
    return IdleEvent(gpu_index=0, started_at_s=started_at_s, mean_util=0.05)


# ---------------------------------------------------------------------------
# Consistency: agent verdict == engine verdict on the same file
# ---------------------------------------------------------------------------

# (fixture filename, expected Verdict) — verdicts are also re-derived from
# the engine in the test body so the assertion fails informatively if the
# engine's own classification of the fixture ever changes.
_FIXTURE_CASES: list[tuple[str, Verdict]] = [
    ("dataloader_starved.json", Verdict.DATALOADER_BOUND),
    ("pcie_bound.json", Verdict.PCIE_BOUND),
    ("checkpoint_bound.json", Verdict.CHECKPOINT_BOUND),
    ("cuda_sync_stalls_v4.json", Verdict.SYNC_BOUND),
    # GPU-only profile (no CPU events): the all-events span is dominated by
    # CUDA-init overhead, so the agent's built Trace must mirror load_trace's
    # GPU-active span or it miscalls this HEALTHY trace UNKNOWN (wall-clock
    # bias). Engine pins this to HEALTHY in test_real_traces.py.
    ("edge_gpu_only.json", Verdict.HEALTHY),
]


@pytest.mark.parametrize("fixture_name,expected_verdict", _FIXTURE_CASES)
def test_file_source_attribution_matches_engine_verdict(
    fixture_name: str, expected_verdict: Verdict
) -> None:
    """For each real fixture, attribute() and diagnose() must agree.

    Path: load_trace(fixture) + diagnose() gives the ground-truth engine
    verdict. Build a FileEventSource(fixture), call attribute() with a
    synthetic IdleEvent (window values are ignored by FileEventSource),
    and assert the agent's verdict equals the engine's verdict.

    The expected_verdict pin is a second check: it catches the case where
    BOTH paths silently agree on the wrong verdict (a fixture-rebuild or
    threshold drift would surface here).
    """
    fixture_path = _FIXTURES_DIR / fixture_name

    engine_trace: Trace = load_trace(fixture_path)
    engine_diag = diagnose(engine_trace)
    assert engine_diag.verdict == expected_verdict, (
        f"engine baseline drift on {fixture_name}: "
        f"got {engine_diag.verdict}, expected {expected_verdict}"
    )

    source = FileEventSource(fixture_path)
    # lookback / now values are deliberately meaningless — FileEventSource
    # ignores the window and returns the entire recorded episode.
    agent_diag = attribute(source, _idle_event(), now_s=0.0, lookback_s=0.0)

    assert agent_diag is not None, (
        f"attribute() returned None on {fixture_name} — "
        f"FileEventSource likely failed to surface events to the engine"
    )
    assert agent_diag.verdict == engine_diag.verdict, (
        f"agent/engine verdict mismatch on {fixture_name}: "
        f"agent={agent_diag.verdict} engine={engine_diag.verdict}"
    )


# ---------------------------------------------------------------------------
# Resilience: bad path must NEVER raise
# ---------------------------------------------------------------------------


def test_file_source_bad_path_returns_empty_and_attribution_none() -> None:
    """Missing / unreadable trace file -> capture() returns []; attribute() -> None.

    Mirrors the daemon-never-crashes invariant: an operator misconfiguring
    ``--trace-file`` should degrade attribution to ``None`` (the same as a
    quiet GPU), not take the agent process down.
    """
    source = FileEventSource("/does/not/exist.json")

    # Capture must not raise; must return [].
    events = source.capture(gpu_index=0, start_s=0.0, end_s=1.0)
    assert events == []

    # Attribute must not raise; must return None (insufficient events).
    diag = attribute(source, _idle_event(), now_s=1.0, lookback_s=0.5)
    assert diag is None


def test_file_source_caches_bad_load_across_calls() -> None:
    """A bad path is only attempted once; subsequent capture()s stay quiet."""
    source = FileEventSource("/does/not/exist.json")
    assert source.capture(gpu_index=0, start_s=0.0, end_s=1.0) == []
    # Second call should not retry the load (and must still not raise).
    assert source.capture(gpu_index=1, start_s=100.0, end_s=200.0) == []


# ---------------------------------------------------------------------------
# Duration derivation: the Trace handed to diagnose() must use the events'
# own span, not the wall-clock window. This is the actual bug-fix anchor.
# ---------------------------------------------------------------------------


def test_built_trace_duration_us_equals_event_span() -> None:
    """The Trace passed into diagnose() has duration_us derived from events.

    Monkeypatches ``gpu_doctor_agent.attribution.diagnose`` to intercept the
    Trace argument, then asserts ``trace.duration_us`` exactly matches
    ``max(ts + dur) - min(ts)`` over the captured events — NOT the
    wall-clock span ``(now_s - window_start_s) * 1e6`` that the agent
    nominally requested.
    """
    fixture_path = _FIXTURES_DIR / "dataloader_starved.json"
    source = FileEventSource(fixture_path)

    captured_trace: dict[str, Trace] = {}

    def _spy(trace: Trace):
        captured_trace["arg"] = trace
        # Return a real diagnosis so attribute() doesn't bail.
        return diagnose(trace)

    # IMPORTANT: patch the binding inside attribution.py (where it was
    # imported), not gpu_doctor_engine.diagnose itself.
    import pytest as _pytest  # local alias keeps the patch scoped

    mp = _pytest.MonkeyPatch()
    mp.setattr(attribution_mod, "diagnose", _spy)
    try:
        # Deliberately choose a window whose wall-clock span (1.0 s = 1e6 us)
        # does NOT match the trace's true span — if duration_us came from
        # the window, this test would catch it.
        diag = attribute(source, _idle_event(started_at_s=0.5), now_s=1.0, lookback_s=0.5)
    finally:
        mp.undo()

    assert diag is not None
    trace = captured_trace.get("arg")
    assert trace is not None, "diagnose() was not called — attribute() bailed early"

    events = source.capture(gpu_index=0, start_s=0.0, end_s=0.0)
    expected_span = max(e.ts + e.dur for e in events) - min(e.ts for e in events)

    assert trace.duration_us > 0
    assert trace.duration_us == expected_span, (
        f"duration_us was {trace.duration_us}, expected event-span "
        f"{expected_span}; wall-clock window span would have been ~1_000_000"
    )
    # And concretely: the wall-clock window we passed in was 1.0 s = 1e6 us.
    # The real fixture spans several seconds, so they MUST differ.
    assert trace.duration_us != 1_000_000


def test_built_trace_duration_us_positive_for_all_fixtures() -> None:
    """Sanity: every fixture-driven attribution yields a positive duration_us.

    A zero duration_us would collapse gpu_utilization to 0.0 (via the
    ``duration_us == 0`` short-circuit in ``Trace.gpu_utilization``) and
    silently disable every util-gated rule in the engine.
    """
    captured: dict[str, Trace] = {}

    def _spy(trace: Trace):
        captured["arg"] = trace
        return diagnose(trace)

    for fixture_name, _ in _FIXTURE_CASES:
        captured.clear()
        source = FileEventSource(_FIXTURES_DIR / fixture_name)

        mp = pytest.MonkeyPatch()
        mp.setattr(attribution_mod, "diagnose", _spy)
        try:
            attribute(source, _idle_event(), now_s=0.0, lookback_s=0.0)
        finally:
            mp.undo()

        trace = captured.get("arg")
        assert trace is not None, f"no diagnose() call for {fixture_name}"
        assert trace.duration_us > 0, (
            f"{fixture_name}: duration_us={trace.duration_us} would zero out util"
        )
