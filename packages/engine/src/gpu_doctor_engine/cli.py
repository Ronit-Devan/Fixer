"""CLI: gpu-doctor [analyze] trace.json"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import click
import typer
import typer.main
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gpu_doctor_engine import __version__, diagnose_with_stats, load_trace
from gpu_doctor_engine.ingest import GPU_ALL_CATS


class _DefaultAnalyzeGroup(typer.main.TyperGroup):
    """Route bare path arguments to the 'analyze' subcommand by default.

    When the first CLI token is not a flag and not a registered subcommand name,
    prepend 'analyze' so that `gpu-doctor trace.json` behaves like
    `gpu-doctor analyze trace.json`.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            args = ["analyze"] + args
        return super().parse_args(ctx, args)


app = typer.Typer(
    help="Diagnose GPU idleness from PyTorch Profiler traces.",
    cls=_DefaultAnalyzeGroup,
)
console = Console()
err_console = Console(stderr=True)


@app.command()
def analyze(
    trace_path: Path = typer.Argument(..., help="Path to PyTorch Profiler trace JSON."),
    json_out: bool = typer.Option(
        False, "--json", help="Emit JSON instead of pretty output."
    ),
    top_events: Annotated[
        int,
        typer.Option("--top-events", help="Print top N longest CPU and GPU events."),
    ] = 0,
    explain: Annotated[
        bool, typer.Option("--explain", help="Print detailed attribution stats.")
    ] = False,
) -> None:
    """Analyze a trace and print the verdict."""
    _run_analyze(trace_path, json_out=json_out, top_events=top_events, explain=explain)


def _run_analyze(
    trace_path: Path,
    *,
    json_out: bool,
    top_events: int,
    explain: bool,
) -> None:
    if not trace_path.exists():
        err_console.print(f"[red]Error:[/red] file not found: {trace_path}")
        raise typer.Exit(1)

    # Always send loading chatter to stderr so --json stdout is clean JSON.
    err_console.print(f"[dim]Loading trace from {trace_path}...[/dim]")
    trace = load_trace(trace_path)
    err_console.print(
        f"[dim]Loaded {len(trace.events)} events, {trace.duration_us / 1000:.0f}ms total[/dim]\n"
    )

    diag, stats = diagnose_with_stats(trace)
    stats["engine_version"] = __version__

    if json_out:
        out = {
            "verdict": diag.verdict.value,
            "confidence": diag.confidence,
            "summary": diag.summary,
            "evidence": diag.evidence,
            "recommended_actions": diag.recommended_actions,
            "metrics": diag.metrics,
        }
        print(json.dumps(out, indent=2))
        return

    color = {
        "healthy": "green",
        "dataloader_bound": "yellow",
        "pcie_bound": "yellow",
        "kernel_launch_bound": "yellow",
        "nccl_bound": "yellow",
        "checkpoint_bound": "yellow",
        "sync_bound": "yellow",
        "unknown": "dim",
    }.get(diag.verdict.value, "white")

    verdict_heading = f"[bold {color}]{diag.verdict.value.upper()}[/bold {color}]"
    if diag.confidence is not None:
        verdict_heading += f"  [dim](confidence: {diag.confidence:.0%})[/dim]"
    console.print(
        Panel(
            f"{verdict_heading}\n\n{diag.summary}",
            title="Verdict",
            border_style=color,
        )
    )

    if diag.evidence:
        table = Table(title="Evidence", show_header=False, border_style="dim")
        for line in diag.evidence:
            table.add_row(f"  {line}")
        console.print(table)

    if diag.recommended_actions:
        console.print("\n[bold]Recommended actions:[/bold]")
        for i, action in enumerate(diag.recommended_actions, 1):
            console.print(f"  {i}. {action}")

    # --explain: print detailed attribution breakdown
    if explain and stats:
        _print_explain(trace, stats)

    # --top-events N: show longest CPU and GPU events
    if top_events > 0:
        _print_top_events(trace, top_events)


def _print_explain(trace, stats: dict) -> None:
    """Print a detailed breakdown of how the verdict was reached."""
    console.print()

    duration_ms = trace.duration_us / 1000
    kernel_ms = trace.gpu_kernel_time_us / 1000
    idle_ms = stats.get("idle_us", 0) / 1000
    memcpy_cat_ms = stats.get("gpu_memcpy_time_us", 0) / 1000
    gpu_active_ms = stats.get("gpu_active_us", 0) / 1000

    tbl = Table(
        title="Explain: attribution breakdown", show_header=True, border_style="dim"
    )
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value")

    tbl.add_row("Total trace duration", f"{duration_ms:.1f} ms")
    tbl.add_row("GPU compute busy (kernels)", f"{kernel_ms:.1f} ms")
    tbl.add_row("GPU memcpy busy (category)", f"{memcpy_cat_ms:.1f} ms")
    tbl.add_row("GPU active total", f"{gpu_active_ms:.1f} ms")
    tbl.add_row("GPU idle (no GPU activity)", f"{idle_ms:.1f} ms")
    tbl.add_row("", "")
    tbl.add_row(
        "DataLoader overlap with idle",
        f"{stats.get('dataloader_us', 0) / 1000:.1f} ms  ({stats.get('dataloader_us', 0) / max(stats.get('idle_us', 1), 1):.0%} of idle)",
    )
    tbl.add_row(
        "NCCL overlap with idle",
        f"{stats.get('nccl_us', 0) / 1000:.1f} ms  ({stats.get('nccl_us', 0) / max(stats.get('idle_us', 1), 1):.0%} of idle)",
    )
    tbl.add_row(
        "Memcpy overlap with idle (name)",
        f"{stats.get('memcpy_name_us', 0) / 1000:.1f} ms",
    )
    tbl.add_row(
        "Memcpy overlap with compute-idle (cat)",
        f"{stats.get('cat_memcpy_us', 0) / 1000:.1f} ms",
    )
    tbl.add_row(
        "Memcpy used (max of above)",
        f"{stats.get('memcpy_us', 0) / 1000:.1f} ms  ({stats.get('memcpy_us', 0) / max(stats.get('idle_us', 1), 1):.0%} of idle)",
    )
    tbl.add_row(
        "GPU memcpy / GPU active ratio",
        f"{stats.get('gpu_memcpy_time_us', 0) / max(stats.get('gpu_active_us', 1), 1):.0%}",
    )
    tbl.add_row(
        "Checkpoint overlap with idle",
        f"{stats.get('checkpoint_us', 0) / 1000:.1f} ms  ({stats.get('checkpoint_us', 0) / max(stats.get('idle_us', 1), 1):.0%} of idle)",
    )
    tbl.add_row(
        "Sync overlap with idle",
        f"{stats.get('sync_us', 0) / 1000:.1f} ms  ({stats.get('sync_us', 0) / max(stats.get('idle_us', 1), 1):.0%} of idle)",
    )
    tbl.add_row("", "")
    avg_k = stats.get("avg_kernel_dur", 0)
    tiny = stats.get("tiny_kernel_ratio", 0)
    tbl.add_row("Average kernel duration", f"{avg_k:.0f} µs")
    tbl.add_row("Tiny kernel ratio (<50µs)", f"{tiny:.0%}")
    tbl.add_row("", "")
    tbl.add_row("Rule fired", stats.get("rule", "—"))

    console.print(tbl)

    hol = stats.get("hol_stats")
    if hol is not None:
        hol_tbl = Table(
            title="Head-of-line blocking analysis",
            show_header=False,
            border_style="dim",
        )
        hol_tbl.add_column("Metric", style="bold")
        hol_tbl.add_column("Value")
        hol_tbl.add_row("DataLoader event count", str(hol["sample_count"]))
        hol_tbl.add_row("Median duration", f"{hol['median_us']:.0f} us")
        hol_tbl.add_row("P99 duration", f"{hol['p99_us']:.0f} us")
        hol_tbl.add_row("HoL ratio (p99/median)", f"{hol['hol_ratio']:.1f}")
        hol_tbl.add_row(
            "HoL blocking likely",
            "[bold red]yes[/bold red]" if hol["hol_blocking_likely"] else "no",
        )
        console.print(hol_tbl)

    decisions = stats.get("decisions")
    if decisions:
        _print_decisions(decisions)


def _print_decisions(decisions: list[dict]) -> None:
    """Render the per-rule decision log produced by diagnose_with_stats."""
    console.print()
    console.print("[bold]Detector decisions:[/bold]")

    max_name_len = max(len(d["rule"]) for d in decisions)
    pad = max_name_len + 2

    for d in decisions:
        rule = d["rule"]
        fired = d["fired"]
        passed = d.get("passed", False)
        value = d["value"]
        threshold = d["threshold"]

        # Three states: a rule can win (fired), pass its own condition but
        # lose the dominant-cause competition (passed), or never trigger.
        if fired:
            marker = "[green]✓[/green]"
            status = "[green]FIRED (won)[/green]"
        elif passed:
            marker = "[yellow]~[/yellow]"
            status = "[yellow]passed, lost competition[/yellow]"
        else:
            marker = "[dim]✗[/dim]"
            status = "[dim]skipped[/dim]"

        rule_padded = rule.ljust(pad)
        console.print(
            f"  {marker} {rule_padded}  value={value:.2f}  threshold={threshold:.2f}   {status}"
        )


def _print_top_events(trace, n: int) -> None:
    """Print the N longest CPU and GPU events."""
    cpu_events = sorted(
        [e for e in trace.events if e.category not in GPU_ALL_CATS],
        key=lambda e: e.dur,
        reverse=True,
    )[:n]

    gpu_events = sorted(
        [e for e in trace.events if e.category in GPU_ALL_CATS],
        key=lambda e: e.dur,
        reverse=True,
    )[:n]

    if cpu_events:
        console.print()
        cpu_tbl = Table(title=f"Top {n} longest CPU events", border_style="dim")
        cpu_tbl.add_column("Name")
        cpu_tbl.add_column("Category")
        cpu_tbl.add_column("Duration (ms)", justify="right")
        for e in cpu_events:
            cpu_tbl.add_row(e.name[:80], e.category, f"{e.dur / 1000:.2f}")
        console.print(cpu_tbl)

    if gpu_events:
        console.print()
        gpu_tbl = Table(title=f"Top {n} longest GPU events", border_style="dim")
        gpu_tbl.add_column("Name")
        gpu_tbl.add_column("Category")
        gpu_tbl.add_column("Duration (ms)", justify="right")
        for e in gpu_events:
            gpu_tbl.add_row(e.name[:80], e.category, f"{e.dur / 1000:.2f}")
        console.print(gpu_tbl)


@app.command()
def report(
    trace_path: Path = typer.Argument(..., help="Path to PyTorch Profiler trace JSON."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output markdown file"
    ),
) -> None:
    """Generate a markdown report for a trace."""
    if not trace_path.exists():
        err_console.print(f"[red]Error:[/red] file not found: {trace_path}")
        raise typer.Exit(1)

    err_console.print(f"[dim]Loading trace from {trace_path}...[/dim]")
    trace = load_trace(trace_path)
    err_console.print(
        f"[dim]Loaded {len(trace.events)} events, {trace.duration_us / 1000:.0f}ms total[/dim]\n"
    )

    diag, stats = diagnose_with_stats(trace)

    md = _build_report(trace_path, trace, diag, stats)

    if output is None:
        sys.stdout.write(md)
    else:
        output.write_text(md, encoding="utf-8")
        err_console.print(f"Report written to {output}")


def _build_report(trace_path: Path, trace, diag, stats: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    verdict_str = diag.verdict.value.upper()
    confidence_line = (
        ""
        if diag.confidence is None
        else f" _(confidence: {diag.confidence:.0%})_"
    )

    duration_ms = trace.duration_us / 1000
    gpu_busy_ms = trace.gpu_kernel_time_us / 1000
    gpu_util_pct = f"{trace.gpu_utilization:.0%}"
    idle_ms = stats.get("idle_us", 0) / 1000
    total_events = len(trace.events)

    dl_ms = stats.get("dataloader_us", 0) / 1000
    nccl_ms = stats.get("nccl_us", 0) / 1000
    memcpy_ms = stats.get("memcpy_us", 0) / 1000
    ckpt_ms = stats.get("checkpoint_us", 0) / 1000

    idle_denom = max(stats.get("idle_us", 0), 1)
    dl_share = f"{stats.get('dataloader_us', 0) / idle_denom:.0%}"
    nccl_share = f"{stats.get('nccl_us', 0) / idle_denom:.0%}"
    memcpy_share = f"{stats.get('memcpy_us', 0) / idle_denom:.0%}"
    ckpt_share = f"{stats.get('checkpoint_us', 0) / idle_denom:.0%}"

    # Evidence lines
    evidence_md = (
        "\n".join(f"- {e}" for e in diag.evidence) if diag.evidence else "- (none)"
    )

    # Recommended actions
    actions_md = (
        "\n".join(f"{i}. {a}" for i, a in enumerate(diag.recommended_actions, 1))
        if diag.recommended_actions
        else "1. No specific actions recommended."
    )

    # Top 10 CPU events
    cpu_events = sorted(
        [e for e in trace.events if e.category not in GPU_ALL_CATS],
        key=lambda e: e.dur,
        reverse=True,
    )[:10]
    gpu_events = sorted(
        [e for e in trace.events if e.category in GPU_ALL_CATS],
        key=lambda e: e.dur,
        reverse=True,
    )[:10]

    def _event_rows(events) -> str:
        if not events:
            return "| (none) | | |\n"
        return "".join(
            f"| {e.name[:80]} | {e.category} | {e.dur / 1000:.2f} ms |\n"
            for e in events
        )

    return f"""\
# GPU Diagnostic Report

**Trace:** `{trace_path.name}`
**Analyzed:** {now}
**Engine:** ET v{__version__}

## Verdict

**{verdict_str}**{confidence_line}

{diag.summary}

## Evidence

{evidence_md}

## Recommended actions

{actions_md}

## Trace statistics

| Metric | Value |
|---|---|
| Total events | {total_events} |
| Trace duration | {duration_ms:.1f} ms |
| GPU busy time | {gpu_busy_ms:.1f} ms |
| GPU utilization | {gpu_util_pct} |
| Total GPU idle | {idle_ms:.1f} ms |

## Bottleneck attribution

| Cause | Idle time overlap | Share of idle |
|---|---|---|
| DataLoader | {dl_ms:.1f} ms | {dl_share} |
| NCCL | {nccl_ms:.1f} ms | {nccl_share} |
| PCIe (Memcpy) | {memcpy_ms:.1f} ms | {memcpy_share} |
| Checkpoint | {ckpt_ms:.1f} ms | {ckpt_share} |

## Top 10 longest CPU events

| Event | Category | Duration |
|---|---|---|
{_event_rows(cpu_events)}\
## Top 10 longest GPU events

| Event | Category | Duration |
|---|---|---|
{_event_rows(gpu_events)}\

---
_Generated by ET v{__version__}. Research foundation: MinatoLoader, eGPU, DataStates-LLM. github.com/devan-p/ET_
"""


if __name__ == "__main__":
    app()
