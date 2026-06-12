# gpu-doctor-engine

The ET diagnostic engine: identify GPU training bottlenecks from a PyTorch
Profiler trace. A trace goes in, a single root-cause verdict comes out —
along with the evidence and the per-rule decision log behind it.

The engine is pure and offline: it reads a Chrome-trace JSON, merges GPU
activity intervals, attributes idle windows to CPU-side causes by name/category
overlap, and returns one of eight verdicts: `HEALTHY`, `DATALOADER_BOUND`,
`PCIE_BOUND`, `KERNEL_LAUNCH_BOUND`, `NCCL_BOUND`, `CHECKPOINT_BOUND`,
`SYNC_BOUND`, `UNKNOWN`.

## Install

```bash
cd packages/engine
uv sync
```

## CLI

```bash
uv run gpu-doctor ../../fixtures/dataloader_starved.json
```

Add `--explain` to see every rule evaluated in order, or `--json` for machine
output:

```bash
uv run gpu-doctor trace.json --explain
uv run gpu-doctor trace.json --json | jq '.verdict'
```

## Library

```python
from gpu_doctor_engine import load_trace, diagnose

diagnosis = diagnose(load_trace("trace.json"))
print(diagnosis.verdict, diagnosis.confidence)
```

`diagnose_with_stats(trace)` returns the same `Diagnosis` plus the decision-log
stats dict used by `--explain`.

## Tests

```bash
uv run pytest
```

## License

MIT.
