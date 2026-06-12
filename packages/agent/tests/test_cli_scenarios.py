"""End-to-end CLI scenario tests.

Drives `gpu-doctor-agent run --mock --scenario ...` through Typer's CliRunner
to assert that the sampler -> buffer -> detector pipeline produces the
expected counted-event behavior for known mock patterns. Asserts on the
"idle_events=N" line in the shutdown summary so the test exercises the same
code path the binary executes (not just the formatted ALERT line).

Scenario playback is hold-last (clamp) — see sampler.MockNvmlSampler. That
makes the event count terminal: once the pattern is exhausted the final
utilization is held forever, so the detector can't re-fire from a late-cycle
dip. The live-loop tests below intentionally pick a LARGE --max-iters so any
regression to cyclic playback (where event counts grow with --max-iters)
would show up immediately.
"""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from gpu_doctor_agent.cli import app
from gpu_doctor_agent.sampler import SCENARIOS, MockNvmlSampler

runner = CliRunner()

_SHUTDOWN_RE = re.compile(r"idle_events=(\d+)")
_ATTRIBUTED_RE = re.compile(r"attributed=(-?\d+)")
_VERDICT_STDOUT_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T[\d:.+Z-]+ gpu=\d+ verdict=SYNC_BOUND"
)


def _idle_events(stdout: str) -> int:
    """Parse the shutdown summary's idle_events counter.

    This is the authoritative source — the binary prints exactly this counter,
    so tests assert on it rather than counting formatted ALERT lines.
    """
    m = _SHUTDOWN_RE.search(stdout)
    assert m is not None, f"no shutdown summary in output:\n---\n{stdout}\n---"
    return int(m.group(1))


def _attributed(stdout: str) -> int:
    """Parse the shutdown summary's attributed counter."""
    m = _ATTRIBUTED_RE.search(stdout)
    assert m is not None, f"no attributed counter in output:\n---\n{stdout}\n---"
    return int(m.group(1))


# -------- sampler-layer scenario plumbing ----------------------------------


def test_scenarios_dict_has_required_keys() -> None:
    for key in ("busy", "idle", "flapping", "recovering"):
        assert key in SCENARIOS
        pattern = SCENARIOS[key]
        assert len(pattern) >= 1
        assert all(0.0 <= u <= 1.0 for u in pattern)


def test_from_scenario_builds_mock_with_named_pattern() -> None:
    s = MockNvmlSampler.from_scenario("idle", gpu_count=2)
    assert s.gpu_count() == 2
    samples = s.sample()
    pattern = SCENARIOS["idle"]
    # Hold-last: gpu g at tick t reads pattern[min(t+g, len-1)].
    assert samples[0].util_pct == pattern[0]
    assert samples[1].util_pct == pattern[min(1, len(pattern) - 1)]


def test_from_scenario_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        MockNvmlSampler.from_scenario("nonexistent")


def test_cli_rejects_unknown_scenario() -> None:
    result = runner.invoke(app, ["run", "--mock", "--scenario", "nope"])
    assert result.exit_code != 0


# -------- --once honors --scenario -----------------------------------------


def test_cli_once_honors_scenario_flapping() -> None:
    """--once must use the scenario sampler, not the default."""
    result = runner.invoke(
        app,
        ["run", "--mock", "--scenario", "flapping", "--once"],
    )
    assert result.exit_code == 0, result.stdout
    # "flapping" starts at 0.05 = 5.0%; "busy" default would be 85.0%.
    assert "5.0" in result.stdout, result.stdout
    assert "85.0" not in result.stdout, result.stdout


# -------- end-to-end live loop: assert on shutdown counter -----------------
# Detector config: idle_threshold=0.20, recovery=0.40, sustain=0.025s.
# Loop config:     interval=0.01s, smoothing = mean of last SMOOTH_SAMPLES=2.
# Sampler playback is hold-last (clamp).
#
# "idle"       pattern = [0.85, 0.85, 0.05*8]; terminal=0.05.
#   smoothed crosses 0.20 around tick 3, sustain elapses ~tick 6 -> CONFIRMED.
#   After playback exhausts, util holds 0.05 — no recovery, no re-fire.
#   Expected: idle_events == 1 for ANY --max-iters past confirmation.
#
# "flapping"   pattern = [0.05, 0.85, 0.05, 0.85, 0.05, 0.85]; terminal=0.85.
#   smoothed alternates 0.45/0.45 (>= recovery), then settles at 0.85.
#   Suspected once at tick 0, recovered tick 1, never sustains again.
#   Expected: idle_events == 0 for ANY --max-iters.
#
# "recovering" pattern = [0.05*8, 0.85, 0.85]; terminal=0.85.
#   Confirms once around tick 3, recovers at tick 8 once smoothed crosses 0.40,
#   then holds 0.85 forever.
#   Expected: idle_events == 1 for ANY --max-iters past confirmation.
#
# "busy"       pattern = [0.85]; terminal=0.85.
#   smoothed always 0.85. Expected: idle_events == 0 for ANY --max-iters.

# Large --max-iters so any regression to cyclic playback would multiply events.
_LARGE_MAX_ITERS = "80"


def _env_for_fast_detector() -> dict[str, str]:
    return {
        "GPU_DOCTOR_IDLE_SUSTAIN_S": "0.025",
        "GPU_DOCTOR_IDLE_UTIL_THRESHOLD": "0.20",
        "GPU_DOCTOR_RECOVERY_UTIL_THRESHOLD": "0.40",
    }


def test_cli_idle_scenario_confirms_exactly_one_event() -> None:
    result = runner.invoke(
        app,
        [
            "run", "--mock", "--scenario", "idle",
            "--interval", "0.01",
            "--max-iters", _LARGE_MAX_ITERS,
        ],
        env=_env_for_fast_detector(),
    )
    assert result.exit_code == 0, result.stdout
    assert _idle_events(result.stdout) == 1, result.stdout


def test_cli_flapping_scenario_emits_no_events() -> None:
    result = runner.invoke(
        app,
        [
            "run", "--mock", "--scenario", "flapping",
            "--interval", "0.01",
            "--max-iters", _LARGE_MAX_ITERS,
        ],
        env=_env_for_fast_detector(),
    )
    assert result.exit_code == 0, result.stdout
    assert _idle_events(result.stdout) == 0, result.stdout


def test_cli_recovering_scenario_confirms_exactly_one_event() -> None:
    result = runner.invoke(
        app,
        [
            "run", "--mock", "--scenario", "recovering",
            "--interval", "0.01",
            "--max-iters", _LARGE_MAX_ITERS,
        ],
        env=_env_for_fast_detector(),
    )
    assert result.exit_code == 0, result.stdout
    assert _idle_events(result.stdout) == 1, result.stdout


def test_cli_busy_scenario_emits_no_events() -> None:
    result = runner.invoke(
        app,
        [
            "run", "--mock",
            "--interval", "0.01",
            "--max-iters", _LARGE_MAX_ITERS,
        ],
        env=_env_for_fast_detector(),
    )
    assert result.exit_code == 0, result.stdout
    assert _idle_events(result.stdout) == 0, result.stdout


def test_cli_idle_event_count_is_max_iters_independent() -> None:
    """Regression guard against cyclic re-firing.

    Under hold-last semantics, the idle scenario must confirm exactly once
    regardless of how long the loop runs — the GPU is stuck idle forever, so
    the detector should never see a recovery that would re-arm it. If sampler
    playback ever wraps to index 0 again, a busy stretch would recover the
    detector and a subsequent idle stretch would fire a second event; running
    long enough would compound that. Asserting equality across two very
    different iteration counts catches that regression.
    """
    counts: list[int] = []
    for max_iters in ("40", "80"):
        result = runner.invoke(
            app,
            [
                "run", "--mock", "--scenario", "idle",
                "--interval", "0.01",
                "--max-iters", max_iters,
            ],
            env=_env_for_fast_detector(),
        )
        assert result.exit_code == 0, result.stdout
        counts.append(_idle_events(result.stdout))
    assert counts == [1, 1], f"idle_events varied with --max-iters: {counts}"


# -------- Tier-2 attribution: end-to-end via CliRunner --------------------
# These tests prove the agent->engine bridge runs in the live loop through
# the SAME code path the shipped binary takes: --attribution-source mock
# builds a MockEventSource(scenario="sync_bound") inside the CLI, which now
# anchors its synthesized events to each capture window. No test-only hook
# is involved — if these pass, the binary attributes.


def test_cli_attribution_none_skips_tier2() -> None:
    """--attribution-source=none skips attribution entirely.

    Prints "attribution pending (Tier 2)" and reports attributed=0.
    """
    result = runner.invoke(
        app,
        [
            "run", "--mock", "--scenario", "idle",
            "--interval", "0.01",
            "--max-iters", _LARGE_MAX_ITERS,
            "--attribution-source", "none",
        ],
        env=_env_for_fast_detector(),
    )
    assert result.exit_code == 0, result.stdout
    assert _idle_events(result.stdout) == 1, result.stdout
    assert _attributed(result.stdout) == 0, result.stdout
    # Legacy Tier-1 alert text must be present.
    assert "attribution pending (Tier 2)" in result.stdout
    # And no verdict tokens should appear in the default-none path.
    assert "SYNC_BOUND" not in result.stdout


def test_cli_idle_scenario_with_attribution_emits_sync_bound() -> None:
    """--scenario idle + --attribution-source mock -> SYNC_BOUND verdict.

    Exercises the real CLI path the binary runs. The CLI's own
    ``MockEventSource(scenario="sync_bound")`` must anchor its synthesized
    events to each capture window so they land inside the live loop's
    monotonic-clock window and clear the MIN_EVENTS guard.
    """
    result = runner.invoke(
        app,
        [
            "run", "--mock", "--scenario", "idle",
            "--interval", "0.01",
            "--max-iters", _LARGE_MAX_ITERS,
            "--attribution-source", "mock",
        ],
        env=_env_for_fast_detector(),
    )
    assert result.exit_code == 0, result.stdout
    # Same idle confirmation count as the Tier-1 baseline test.
    assert _idle_events(result.stdout) == 1, result.stdout
    # The bridge ran and produced a verdict for that confirmation.
    assert _attributed(result.stdout) >= 1, result.stdout
    # The alert line carries the engine's verdict, not the Tier-1 fallback.
    assert "SYNC_BOUND" in result.stdout, result.stdout
    assert "attribution pending (Tier 2)" not in result.stdout


def test_cli_busy_scenario_with_attribution_yields_no_attributions() -> None:
    """No idle confirmations means no attributions even with a source wired in."""
    result = runner.invoke(
        app,
        [
            "run", "--mock",  # busy scenario (default)
            "--interval", "0.01",
            "--max-iters", _LARGE_MAX_ITERS,
            "--attribution-source", "mock",
        ],
        env=_env_for_fast_detector(),
    )
    assert result.exit_code == 0, result.stdout
    assert _idle_events(result.stdout) == 0, result.stdout
    assert _attributed(result.stdout) == 0, result.stdout


def test_demo_mode_idle_confirmed_prints_verdict_to_stdout() -> None:
    """--demo-mode: MockNvml idle + MockEventSource -> verdict on stdout."""
    result = runner.invoke(
        app,
        ["run", "--demo-mode", "--max-iters", "20"],
    )
    assert result.exit_code == 0, result.stdout
    assert "attribution=mock" in result.stdout, result.stdout
    assert "trace_file=" not in result.stdout, result.stdout
    assert _idle_events(result.stdout) == 1, result.stdout
    assert _attributed(result.stdout) >= 1, result.stdout
    assert _VERDICT_STDOUT_RE.search(result.stdout), result.stdout
    assert "SYNC_BOUND" in result.stdout, result.stdout


def test_cli_attribution_file_uses_bundled_trace_when_omitted() -> None:
    """--attribution-source=file without --trace-file resolves the bundled default."""
    result = runner.invoke(
        app,
        [
            "run", "--mock", "--scenario", "idle",
            "--interval", "0.01",
            "--max-iters", _LARGE_MAX_ITERS,
            "--attribution-source", "file",
        ],
        env=_env_for_fast_detector(),
    )
    assert result.exit_code == 0, result.stdout
    assert _idle_events(result.stdout) == 1, result.stdout
    assert _attributed(result.stdout) >= 1, result.stdout


def test_cli_attribution_rejects_unknown_source() -> None:
    result = runner.invoke(
        app,
        [
            "run", "--mock",
            "--attribution-source", "garbage",
        ],
    )
    assert result.exit_code != 0


def test_cli_attribution_engine_crash_falls_back_to_plain_alert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An engine.diagnose() failure must NOT take the daemon down.

    The loop logs and emits the Tier-1 fallback line, but keeps sampling and
    exits cleanly. attributed stays at 0 since no verdict was produced.
    """
    # Monkeypatch the diagnose symbol the attribution module imported.
    from gpu_doctor_agent import attribution as attribution_mod

    def _boom(_trace: object) -> None:
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(attribution_mod, "diagnose", _boom)

    result = runner.invoke(
        app,
        [
            "run", "--mock", "--scenario", "idle",
            "--interval", "0.01",
            "--max-iters", _LARGE_MAX_ITERS,
            "--attribution-source", "mock",
        ],
        env=_env_for_fast_detector(),
    )

    assert result.exit_code == 0, result.stdout
    assert _idle_events(result.stdout) == 1, result.stdout
    assert _attributed(result.stdout) == 0, result.stdout
    # Fall back to the plain "attribution pending" line — daemon survived.
    assert "attribution pending (Tier 2)" in result.stdout
