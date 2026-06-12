# Installing gpu-doctor

## From source (recommended)

```bash
git clone https://github.com/devan-p/ET
cd ET/packages/engine
uv sync
uv run gpu-doctor --help
```

## From wheel

```bash
pip install gpu_doctor_engine-0.3.0-py3-none-any.whl
gpu-doctor /path/to/trace.json
```

## Requirements

- Python 3.10+
- A PyTorch Profiler Chrome trace JSON file (export with `prof.export_chrome_trace`)
