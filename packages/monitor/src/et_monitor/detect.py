"""One-command setup: look at the running llama.cpp box and figure out its
decode roofline, so ET can answer "40% of what?" from the first tick.

``et-monitor detect`` (or ``--detect``) probes the live server and the GPU and
writes a ``WorkloadSpec`` to ``~/.et/workload.json``:

  * llama-server ``/props``  -> model path, context size, n_gpu_layers (if exposed)
  * the model's GGUF header  -> layer count + on-disk size (~ resident weight bytes)
  * the GPU name             -> memory bandwidth (lookup table; operator-overridable)

It then prints the single-stream ceiling and (if the server is serving) where the
box sits against it. Everything is best-effort: anything it can't detect is left
None and the monitor degrades to its util-based heuristics.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Sequence

from et_monitor.llama import LlamaScraper
from et_monitor.perf import (
    WorkloadSpec,
    bandwidth_for,
    default_spec_path,
    read_gguf_metadata,
    roofline,
)


def discover_llama_url(
    *,
    host: str = "localhost",
    ports: Sequence[int] = (8080, 8081, 8000, 8888),
    timeout_s: float = 0.5,
) -> str | None:
    """Return the first ``http://host:port`` whose ``/metrics`` answers, else None."""
    for port in ports:
        url = f"http://{host}:{port}"
        try:
            req = urllib.request.Request(f"{url}/metrics", headers={"Accept": "text/plain"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                if resp.status == 200:
                    return url
        except (urllib.error.URLError, OSError, ValueError):
            continue
    return None


def build_workload_spec(
    *,
    gpu_name: str | None,
    llama_url: str | None = None,
    model_path: str | None = None,
    n_gpu_layers: int | None = None,
    mem_bandwidth_gb_s: float | None = None,
    timeout_s: float = 2.0,
) -> tuple[WorkloadSpec, list[str]]:
    """Assemble a WorkloadSpec from whatever the box exposes. Returns (spec, notes)
    where ``notes`` explains, line by line, what was detected and what's missing.
    """
    notes: list[str] = []
    model_name: str | None = None
    n_layers: int | None = None
    model_bytes: float | None = None

    # 1) Ask the live server for its launch config.
    if llama_url:
        props = LlamaScraper(llama_url, timeout_s=timeout_s).read_props()
        if props.reachable:
            notes.append(f"llama-server: reachable at {llama_url}")
            model_path = model_path or props.model_path
            if n_gpu_layers is None and props.n_gpu_layers is not None:
                n_gpu_layers = props.n_gpu_layers
        else:
            notes.append(f"llama-server: /props not reachable at {llama_url} (ok)")

    # 2) Read the GGUF header for layer count + size (the roofline's denominator).
    if model_path and Path(model_path).is_file():
        info = read_gguf_metadata(model_path)
        if info is not None:
            model_bytes = float(info.file_bytes)
            n_layers = info.n_layers
            model_name = info.name or Path(model_path).stem
            notes.append(
                f"model: {model_name or model_path} "
                f"({model_bytes / 1e9:.1f} GB, {n_layers or '?'} layers)"
            )
        else:
            notes.append(f"model: {model_path} is not a readable GGUF (ok)")
    elif model_path:
        notes.append(f"model: {model_path} not found on this host (pass --model)")
    else:
        notes.append("model: unknown - pass --model <path.gguf> for the roofline")

    # 3) Resolve GPU memory bandwidth from the card name (override-able).
    if mem_bandwidth_gb_s is None and gpu_name:
        mem_bandwidth_gb_s = bandwidth_for(gpu_name)
    if mem_bandwidth_gb_s:
        notes.append(f"GPU: {gpu_name or '?'} (~{mem_bandwidth_gb_s:.0f} GB/s)")
    else:
        notes.append(
            f"GPU: {gpu_name or '?'} - bandwidth unknown; pass "
            "--gpu-bandwidth <GB/s> to enable MBU/ceiling"
        )

    spec = WorkloadSpec(
        model_bytes=model_bytes,
        n_layers=n_layers,
        n_gpu_layers=n_gpu_layers,
        mem_bandwidth_gb_s=mem_bandwidth_gb_s,
        model_name=model_name,
        gpu_name=gpu_name,
    )
    return spec, notes


def roofline_preview(
    spec: WorkloadSpec,
    gen_tok_s: float | None = None,
    *,
    gpu_mem_total_mb: float | None = None,
) -> list[str]:
    """Human-readable lines describing the spec's single-stream ceiling."""
    lines: list[str] = []
    # Catch a model that can't even load BEFORE llama-server OOMs at runtime.
    if spec.model_bytes and gpu_mem_total_mb:
        model_gb = spec.model_bytes / 1e9
        vram_gb = gpu_mem_total_mb / 1024
        if model_gb > vram_gb * 0.9:
            lines.append(
                f"WARNING: model (~{model_gb:.1f} GB) is too big for this GPU's VRAM "
                f"(~{vram_gb:.0f} GB) -- llama-server will OOM. Use a smaller quant "
                "(e.g. Q4_K_M) or a bigger card."
            )
    if not spec.has_roofline:
        lines.append(
            "Not enough info for a roofline yet (need model size + GPU bandwidth)."
        )
        if spec.offload_fraction < 0.98:
            lines.append(
                f"  But -ngl puts only {spec.offload_fraction:.0%} of the model on the "
                "GPU - that alone caps throughput. Raise it to 999."
            )
        return lines
    rl = roofline(spec, gen_tok_s, concurrency=1.0)
    assert rl is not None
    lines.append(f"Single-stream decode ceiling: ~{rl.ceiling_tok_s:.0f} tok/s")
    if rl.partial_offload:
        lines.append(
            f"  WARNING: only {rl.offload_fraction:.0%} of the model is on the GPU "
            "(-ngl too low). Fix this first - it caps everything."
        )
    if gen_tok_s and rl.throughput_pct is not None:
        lines.append(
            f"  You are at ~{gen_tok_s:.0f} tok/s = {rl.throughput_pct:.0%} of the "
            f"achievable ceiling (MBU {rl.mbu:.0%})."
        )
        if rl.at_bandwidth_wall:
            lines.append(
                "  That is near the physical single-stream wall - utilization can't "
                "hit 90% at concurrency 1. Batch concurrent requests to go higher."
            )
        else:
            lines.append(
                "  There is real headroom below the wall: ET will attribute the gap "
                "(offload / host-bound / under-batching) and recommend the fix."
            )
    return lines


def run_detect(
    *,
    gpu_name: str | None,
    llama_url: str | None,
    model_path: str | None = None,
    n_gpu_layers: int | None = None,
    mem_bandwidth_gb_s: float | None = None,
    gen_tok_s: float | None = None,
    gpu_mem_total_mb: float | None = None,
    save_path: str | Path | None = None,
    print_fn: Callable[[str], None] = print,
) -> WorkloadSpec:
    """Detect, print a preview, and persist the spec. Returns the spec."""
    if llama_url is None:
        llama_url = discover_llama_url()
        if llama_url:
            print_fn(f"  discovered llama-server at {llama_url}")
    spec, notes = build_workload_spec(
        gpu_name=gpu_name, llama_url=llama_url, model_path=model_path,
        n_gpu_layers=n_gpu_layers, mem_bandwidth_gb_s=mem_bandwidth_gb_s,
    )
    print_fn("\nET workload detection")
    print_fn("=====================")
    for n in notes:
        print_fn(f"  {n}")
    print_fn("")
    for line in roofline_preview(spec, gen_tok_s, gpu_mem_total_mb=gpu_mem_total_mb):
        print_fn(line)
    path = Path(save_path) if save_path else default_spec_path()
    spec.save(path)
    print_fn(f"\nSaved -> {path}\n(et-monitor will use this automatically.)")
    return spec
