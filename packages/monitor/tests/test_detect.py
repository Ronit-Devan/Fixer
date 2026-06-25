"""Auto-setup: build a WorkloadSpec from GGUF + GPU name and preview the ceiling."""

from __future__ import annotations

import struct

from et_monitor.detect import build_workload_spec, roofline_preview, run_detect


def _gguf(tmp_path, n_layers=32, size_pad=4_000_000):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    data = (
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 2)
        + s("general.architecture") + struct.pack("<I", 8) + s("llama")
        + s("llama.block_count") + struct.pack("<I", 4) + struct.pack("<I", n_layers)
    )
    data += b"\x00" * size_pad  # pad so file size ~ a believable model size
    p = tmp_path / "model.gguf"
    p.write_bytes(data)
    return p


def test_build_spec_from_gguf_and_known_gpu(tmp_path):
    model = _gguf(tmp_path, n_layers=32)
    spec, notes = build_workload_spec(
        gpu_name="NVIDIA RTX PRO 4000 Blackwell", llama_url=None,
        model_path=str(model), n_gpu_layers=16,
    )
    assert spec.n_layers == 32
    assert spec.n_gpu_layers == 16
    assert spec.offload_fraction == 0.5
    assert spec.mem_bandwidth_gb_s == 672.0
    assert spec.has_roofline
    assert any("layers" in n for n in notes)


def test_build_spec_unknown_gpu_has_no_bandwidth(tmp_path):
    model = _gguf(tmp_path)
    spec, notes = build_workload_spec(
        gpu_name="Mystery Card 9000", llama_url=None, model_path=str(model),
    )
    assert spec.mem_bandwidth_gb_s is None
    assert any("bandwidth unknown" in n for n in notes)


def test_preview_warns_on_partial_offload(tmp_path):
    model = _gguf(tmp_path, n_layers=32)
    spec, _ = build_workload_spec(
        gpu_name="RTX PRO 4000 Blackwell", llama_url=None,
        model_path=str(model), n_gpu_layers=10,
    )
    lines = "\n".join(roofline_preview(spec, gen_tok_s=15.0))
    assert "ceiling" in lines.lower()
    assert "only" in lines.lower() and "%" in lines  # partial-offload warning


def test_preview_warns_when_model_too_big_for_vram():
    # 8 GB model on a 6 GB card -> warn at detect time, before llama-server OOMs.
    from et_monitor.perf import WorkloadSpec

    spec = WorkloadSpec(model_bytes=8e9, n_layers=32, n_gpu_layers=32, mem_bandwidth_gb_s=672.0)
    lines = "\n".join(roofline_preview(spec, gpu_mem_total_mb=6144))
    assert "too big" in lines.lower() and "oom" in lines.lower()
    # ...but not when it fits.
    ok = "\n".join(roofline_preview(spec, gpu_mem_total_mb=24576))
    assert "too big" not in ok.lower()


def test_run_detect_persists_spec(tmp_path):
    model = _gguf(tmp_path)
    out_lines: list[str] = []
    save = tmp_path / "workload.json"
    spec = run_detect(
        gpu_name="RTX PRO 4000 Blackwell", llama_url=None, model_path=str(model),
        n_gpu_layers=32, save_path=save, print_fn=out_lines.append,
    )
    assert save.is_file()
    assert spec.has_roofline
    from et_monitor.perf import WorkloadSpec

    assert WorkloadSpec.load_or_none(save) == spec
    assert any("ceiling" in ln.lower() for ln in out_lines)
