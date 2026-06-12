"""CLI for the agent sampling spine.

Single command, `run`, that drives the sampler -> buffer -> detector loop
with a monotonic-clock scheduler. Signals are wired for clean K8s shutdown.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Annotated, Callable

import typer
from rich.console import Console
from rich.table import Table

from gpu_doctor_agent.attribution import (
    attribute,
    format_attributed_alert,
    format_verdict_stdout,
)
from gpu_doctor_agent.buffer import RingBuffer
from gpu_doctor_agent.config import AgentConfig, ConfigError
from gpu_doctor_agent.detector import IdleDetector, IdleEvent
from gpu_doctor_agent.events import EventSource, FileEventSource, MockEventSource
from gpu_doctor_agent.sampler import SCENARIOS, Sample, Sampler, get_sampler

# Hidden test hook. When non-None, used as the live event source regardless
# of --attribution-source. Tests set this to inject a custom EventSource
# (e.g. one that anchors synthesized events to the actual capture window).
# Production code never reads it after startup.
_TEST_EVENT_SOURCE: EventSource | None = None

log = logging.getLogger("gpu_doctor_agent")

# Detector input is the mean of the last SMOOTH_SAMPLES samples — just enough
# to absorb single-sample noise without diluting a real dip with stale busy
# samples. The IdleDetector's idle_sustain_s timer is the ONLY place sustain
# is enforced; if we fed it a mean over the sustain window too, we'd be
# double-counting sustain and the threshold would almost never cross.
SMOOTH_SAMPLES: int = 2

app = typer.Typer(
    help="ET live GPU observability agent (sampling spine + idle detection).",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


@app.callback()
def _root() -> None:
    """Keep `run` as an explicit subcommand even though it's the only one,
    so the namespace is stable when more commands are added later."""


def _render_samples_table(samples: list[Sample]) -> Table:
    t = Table(title="GPU samples")
    t.add_column("GPU", justify="right")
    t.add_column("util %", justify="right")
    t.add_column("mem used (MB)", justify="right")
    t.add_column("mem total (MB)", justify="right")
    t.add_column("SM MHz", justify="right")
    t.add_column("power (W)", justify="right")
    for s in samples:
        t.add_row(
            str(s.gpu_index),
            f"{s.util_pct * 100:.1f}" if s.util_pct is not None else "—",
            f"{s.mem_used_mb:.0f}" if s.mem_used_mb is not None else "—",
            f"{s.mem_total_mb:.0f}" if s.mem_total_mb is not None else "—",
            str(s.sm_clock_mhz) if s.sm_clock_mhz is not None else "—",
            f"{s.power_w:.1f}" if s.power_w is not None else "—",
        )
    return t


def _format_idle_alert(event: IdleEvent, sustain_s: float) -> str:
    pct = event.mean_util * 100
    return (
        f"[bold red]ALERT[/] GPU {event.gpu_index} idle for {sustain_s:g}s "
        f"at {pct:.1f}% — attribution pending (Tier 2)"
    )


class _ShutdownFlag:
    """Tiny mutable flag so signal handlers can poke the loop."""

    __slots__ = ("triggered", "signal_name")

    def __init__(self) -> None:
        self.triggered = False
        self.signal_name: str | None = None

    def trip(self, name: str) -> None:
        self.triggered = True
        self.signal_name = name


def _install_signal_handlers(flag: _ShutdownFlag) -> None:
    def _handler(signum: int, _frame: object) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        flag.trip(name)

    # Some test runners disallow installing handlers off the main thread;
    # tolerate that silently — tests bypass run_loop entirely.
    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError):
        pass


def run_loop(
    config: AgentConfig,
    sampler: Sampler,
    *,
    max_iters: int = 0,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], float] = time.monotonic,
    on_event: Callable[[IdleEvent], None] | None = None,
    shutdown: _ShutdownFlag | None = None,
    event_source: EventSource | None = None,
    lookback_s: float = 5.0,
) -> tuple[int, int, int]:
    """Main sampling loop. Returns (total_samples, total_idle_events, total_attributed).

    `max_iters=0` means run forever (until signal). Time math is monotonic.
    The scheduler measures work time per tick and subtracts it from the
    sleep so the loop holds its target rate instead of drifting outward.

    When ``event_source`` is provided, each confirmed ``IdleEvent`` triggers a
    call to ``attribute()`` to bridge into the engine for a verdict. A failure
    inside attribution is logged and the loop falls back to the plain Tier-1
    alert — attribution never kills the daemon.
    """
    # Buffers and detectors are created lazily on first sight of a gpu_index.
    # Keeps startup zero-cost and tolerates a GPU appearing mid-run.
    buffers: dict[int, RingBuffer] = {}
    detectors: dict[int, IdleDetector] = {}

    total_samples = 0
    total_events = 0
    total_attributed = 0
    iters = 0

    while True:
        if shutdown is not None and shutdown.triggered:
            break
        if max_iters and iters >= max_iters:
            break

        tick_start = now_fn()
        try:
            samples = sampler.sample()
        except Exception as e:
            # A sampler-level failure (not a per-GPU one — those are isolated
            # inside the sampler) is logged and the loop continues. The daemon
            # must not die on a transient.
            log.exception("sampler failure on tick %d: %s", iters, e)
            samples = []

        for s in samples:
            buf = buffers.get(s.gpu_index)
            if buf is None:
                buf = RingBuffer(config.ring_capacity)
                buffers[s.gpu_index] = buf
                detectors[s.gpu_index] = IdleDetector(
                    s.gpu_index, config, now_fn=now_fn
                )
            buf.append(s)
            total_samples += 1

            # Feed the detector a short-window smoothed util — NOT a sustain-
            # window mean. Sustain is the detector's job; using the sustain
            # window here would mix busy samples into the mean and prevent
            # the threshold from ever crossing. recent(SMOOTH_SAMPLES) gives
            # just enough smoothing to ignore a single noisy sample.
            recent_samples = buf.recent(SMOOTH_SAMPLES)
            valid_utils = [
                rs.util_pct for rs in recent_samples if rs.util_pct is not None
            ]
            smoothed: float | None = (
                sum(valid_utils) / len(valid_utils) if valid_utils else None
            )
            event = detectors[s.gpu_index].observe(smoothed, now=s.timestamp_s)
            if event is not None:
                total_events += 1
                if on_event is not None:
                    on_event(event)
                elif event_source is None:
                    # No-attribution path — preserves byte-identical behavior
                    # with pre-Tier-2 builds when --attribution-source=none.
                    console.print(_format_idle_alert(event, config.idle_sustain_s))
                else:
                    # Tier-2 attribution path. Anything here must be defensive:
                    # an attribution failure falls back to the plain alert and
                    # the daemon keeps sampling.
                    diagnosis = None
                    try:
                        diagnosis = attribute(
                            event_source,
                            event,
                            now_s=s.timestamp_s,
                            lookback_s=lookback_s,
                        )
                    except Exception:
                        log.exception(
                            "attribute() raised; falling back to plain alert"
                        )
                    if diagnosis is not None:
                        total_attributed += 1
                        print(
                            format_verdict_stdout(
                                event.gpu_index, diagnosis
                            ),
                            flush=True,
                        )
                    console.print(
                        format_attributed_alert(
                            event, diagnosis, config.idle_sustain_s
                        )
                    )

        iters += 1

        # Monotonic-clock scheduler: subtract work time so loop holds its rate.
        # If sampling overran the budget, run the next tick immediately rather
        # than sleeping a negative duration.
        elapsed = now_fn() - tick_start
        sleep_s = config.sample_interval_s - elapsed
        if sleep_s > 0:
            # Wake early if a signal arrives mid-sleep by chunking the sleep
            # only when the deadline is long. For typical 1s intervals just
            # sleep through; the signal will be observed on the next iteration.
            sleep_fn(sleep_s)

    return total_samples, total_events, total_attributed


_ATTRIBUTION_SOURCES = ("none", "mock", "file")

# Bundled trace for demos and the default file-replay path (no GPU required).
_AGENT_PKG_DIR = Path(__file__).resolve().parent
_AGENT_ROOT = _AGENT_PKG_DIR.parent.parent  # packages/agent
_REPO_ROOT = _AGENT_ROOT.parent.parent  # ET repo root (packages/agent -> packages -> root)
_DEFAULT_TRACE_CANDIDATES: tuple[Path, ...] = (
    _REPO_ROOT / "fixtures" / "cuda_sync_stalls_v4.json",
    _REPO_ROOT / "fixtures" / "cuda_sync_stalls.json",
)


def _find_default_trace_file() -> Path:
    """Locate a recorded trace for automatic Tier-2 file attribution."""
    for candidate in _DEFAULT_TRACE_CANDIDATES:
        if candidate.is_file():
            return candidate
    here = Path(__file__).resolve()
    for parent in here.parents:
        fixture = parent / "fixtures" / "cuda_sync_stalls_v4.json"
        if fixture.is_file():
            return fixture
    raise typer.BadParameter(
        "no default trace file found; set GPU_DOCTOR_TRACE_FILE or pass --trace-file"
    )


def _resolve_trace_file(trace_file: Path | None) -> Path:
    if trace_file is not None:
        return trace_file
    env_path = os.environ.get("GPU_DOCTOR_TRACE_FILE", "").strip()
    if env_path:
        return Path(env_path)
    return _find_default_trace_file()


def _build_event_source(
    attribution_source: str, trace_file: Path | None
) -> EventSource | None:
    """Construct the event source for the run loop.

    Precedence is deliberate:
      1. ``--attribution-source=none`` always returns ``None``, ignoring the
         test hook. This keeps the legacy pre-Tier-2 codepath byte-identical
         to its old behavior — a test hook left over from another suite must
         not silently enable attribution when the user asked for none.
      2. For any other mode, ``_TEST_EVENT_SOURCE`` (if set) wins, letting
         tests substitute a window-anchored mock for the static default.
      3. Otherwise build the appropriate default source for the mode.
    """
    if attribution_source == "none":
        return None
    if _TEST_EVENT_SOURCE is not None:
        return _TEST_EVENT_SOURCE
    if attribution_source == "mock":
        # Default mock: a sync-bound demo. Tests that need a different shape
        # inject via ``_TEST_EVENT_SOURCE`` rather than racing this default.
        return MockEventSource(scenario="sync_bound")
    if attribution_source == "file":
        return FileEventSource(_resolve_trace_file(trace_file))
    raise typer.BadParameter(
        f"unknown --attribution-source {attribution_source!r}; "
        f"valid: {_ATTRIBUTION_SOURCES}"
    )


@app.command()
def run(
    mock: Annotated[
        bool,
        typer.Option("--mock/--no-mock", help="Use the MockNvmlSampler instead of NVML."),
    ] = False,
    interval: Annotated[
        float | None,
        typer.Option(
            "--interval",
            help="Override sample_interval_s (seconds between ticks).",
        ),
    ] = None,
    once: Annotated[
        bool,
        typer.Option(
            "--once",
            help="Take a single sample set, print as a table, and exit (smoke test).",
        ),
    ] = False,
    max_iters: Annotated[
        int,
        typer.Option(
            "--max-iters",
            help="Stop after N loops. 0 = run forever.",
        ),
    ] = 0,
    scenario: Annotated[
        str,
        typer.Option(
            "--scenario",
            help=(
                "Mock utilization scenario (one of: "
                + ", ".join(sorted(SCENARIOS))
                + "). Used when --mock is set or NVML is unavailable."
            ),
        ),
    ] = "busy",
    demo_mode: Annotated[
        bool,
        typer.Option(
            "--demo-mode",
            help=(
                "Mac-friendly demo (no GPU): MockNvmlSampler scenario=idle, "
                "MockEventSource Tier-2 attribution, fast detector. "
                "Use --max-iters 20 for a finite run."
            ),
        ),
    ] = False,
    attribution_source: Annotated[
        str,
        typer.Option(
            "--attribution-source",
            help=(
                "Tier-2 attribution event source. "
                "'file' (default) replays a profiler trace via FileEventSource on "
                "each IDLE_CONFIRMED transition. "
                "'mock' uses an in-memory synthetic source. "
                "'none' skips attribution (Tier-1 alert only)."
            ),
        ),
    ] = "file",
    trace_file: Annotated[
        Path | None,
        typer.Option(
            "--trace-file",
            help=(
                "Chrome-trace JSON for file attribution "
                "(default: bundled demo trace or GPU_DOCTOR_TRACE_FILE)."
            ),
        ),
    ] = None,
) -> None:
    """Run the sampling spine."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if scenario not in SCENARIOS:
        err_console.print(
            f"[red]unknown scenario {scenario!r}; valid: {sorted(SCENARIOS)}[/]"
        )
        raise typer.Exit(code=2)

    if attribution_source not in _ATTRIBUTION_SOURCES:
        err_console.print(
            f"[red]unknown --attribution-source {attribution_source!r}; "
            f"valid: {list(_ATTRIBUTION_SOURCES)}[/]"
        )
        raise typer.Exit(code=2)

    if demo_mode:
        mock = True
        scenario = "idle"
        if attribution_source == "none":
            err_console.print(
                "[red]--demo-mode requires Tier-2 attribution; "
                "omit --attribution-source=none[/]"
            )
            raise typer.Exit(code=2)
        # Demo always uses in-memory MockEventSource (no trace file on disk).
        attribution_source = "mock"

    try:
        config = AgentConfig.from_env()
        if interval is not None:
            config = config.with_overrides(sample_interval_s=interval)
        if demo_mode:
            config = config.with_overrides(
                sample_interval_s=interval if interval is not None else 0.01,
                idle_sustain_s=0.025,
            )
    except ConfigError as e:
        err_console.print(f"[red]config error:[/] {e}")
        raise typer.Exit(code=2) from e

    # Sampler is constructed once with the requested scenario; --once reuses
    # this same instance so scenario selection is honored in single-shot mode.
    sampler = get_sampler(config, mock=mock, scenario=scenario)

    if once:
        samples = sampler.sample()
        console.print(_render_samples_table(samples))
        return

    try:
        event_source = _build_event_source(attribution_source, trace_file)
    except typer.BadParameter as e:
        err_console.print(f"[red]{e}[/]")
        raise typer.Exit(code=2) from e

    shutdown = _ShutdownFlag()
    _install_signal_handlers(shutdown)

    mode_note = " [demo]" if demo_mode else ""
    console.print(
        f"[dim]Starting agent{mode_note}: interval={config.sample_interval_s}s "
        f"idle_threshold={config.idle_util_threshold} "
        f"sustain={config.idle_sustain_s}s "
        f"recovery={config.recovery_util_threshold} "
        f"attribution={attribution_source}[/]"
    )
    if attribution_source == "file" and not demo_mode:
        try:
            resolved_trace = _resolve_trace_file(trace_file)
        except typer.BadParameter as e:
            err_console.print(f"[red]{e}[/]")
            raise typer.Exit(code=2) from e
        console.print(f"[dim]trace_file={resolved_trace}[/]")

    try:
        total_samples, total_events, total_attributed = run_loop(
            config,
            sampler,
            max_iters=max_iters,
            shutdown=shutdown,
            event_source=event_source,
        )
    except KeyboardInterrupt:
        # In case SIGINT slips past the handler install (e.g., during startup).
        total_samples = total_events = total_attributed = -1

    suffix = (
        f" (signal: {shutdown.signal_name})" if shutdown.signal_name else ""
    )
    console.print(
        f"[dim]Shutdown{suffix}. samples={total_samples} "
        f"idle_events={total_events} attributed={total_attributed}[/]"
    )


def main() -> None:  # pragma: no cover - thin wrapper
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
