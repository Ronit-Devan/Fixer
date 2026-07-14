# CLAUDE.md

## ⏸️ ACTIVE TASK — RESUME HERE if work is in progress

There is an autonomous multi-phase task running on branch **`client-sff-optimization`**:
optimizing ET for a specific client (RTX PRO 4000 Blackwell **SFF**, llama.cpp, a
Qwen MoE served at fixed quant), targeting max effective tok/s without changing the
model/quant or hurting quality.

**To continue it:**
1. `git checkout client-sff-optimization && git pull`
2. Read **`docs/notes/client-sff-optimization-progress.md`** in full — it holds the
   mission, the client benchmark analysis, key repo facts, and a per-phase checklist.
3. Resume at the first unchecked box; commit **and push** after each logical unit.

If no unchecked boxes remain, the task is done — see the summary at the end of that file.

## Repo orientation

ET attributes GPU idleness to a root cause. Packages under `packages/`:
- `engine` — diagnoses a PyTorch Profiler trace (training-side). CLI `gpu-doctor`.
- `monitor` — live llama.cpp serving monitor + decode **roofline** (MBU / single-stream
  tok/s ceiling / partial-offload). Most client-optimization work lives here.
- `remediation` — safe actuation layer (auto-applies non-disruptive fixes, human-gates
  disruptive ones; never kills the running workload).
- `agent` — eBPF/torch live-capture agent. `web` — API + frontend.

Each package uses `uv`. Test a package: `cd packages/<pkg> && uv run pytest`.
Lint/type: `uv run ruff check src/ tests/` and (where configured) `uv run mypy src/`.
