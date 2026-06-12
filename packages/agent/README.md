# gpu-doctor-agent

Always-on sampling spine for ET's live GPU observability agent (Product A).

This package provides the low-overhead per-GPU sampling loop and the idle-detection
state machine. Higher-tier attribution (eBPF, CUPTI, PTX, Kubernetes integration)
is intentionally **not** part of this package — it is plugged in at the `IdleEvent`
boundary surfaced by `detector.IdleDetector`.

## Quick start

```bash
uv run gpu-doctor-agent run --mock --once
```

Drop `--mock` on a machine with an NVIDIA GPU and `pynvml` installed to sample
real devices via NVML.
