"""Tests for the decode roofline + GGUF metadata reader (perf.py)."""

from __future__ import annotations

import struct

from et_monitor.perf import (
    WorkloadSpec,
    bandwidth_for,
    read_gguf_metadata,
    roofline,
)


# -- bandwidth lookup --------------------------------------------------------


def test_bandwidth_longest_substring_wins():
    # "rtx pro 4000 blackwell" must beat the generic "blackwell" fallback.
    bw = bandwidth_for("NVIDIA RTX PRO 4000 Blackwell SFF")
    assert bw == 672.0
    assert bandwidth_for("Tesla H100 PCIe") == 3350.0
    assert bandwidth_for("Some Unknown Card 9000") is None
    assert bandwidth_for(None) is None


# -- GGUF reader -------------------------------------------------------------


def _gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _make_gguf(kvs: list[tuple[str, int, bytes]]) -> bytes:
    # header: magic, version, tensor_count, kv_count
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kvs))
    for key, vtype, val in kvs:
        out += _gguf_string(key) + struct.pack("<I", vtype) + val
    return out


def test_read_gguf_metadata_extracts_layers_and_arch(tmp_path):
    data = _make_gguf(
        [
            ("general.architecture", 8, _gguf_string("llama")),
            ("general.name", 8, _gguf_string("Qwen2.5 7B Instruct")),
            ("llama.block_count", 4, struct.pack("<I", 32)),
        ]
    )
    p = tmp_path / "model.gguf"
    p.write_bytes(data)
    info = read_gguf_metadata(p)
    assert info is not None
    assert info.architecture == "llama"
    assert info.n_layers == 32
    assert info.file_bytes == len(data)


def test_read_gguf_metadata_skips_arrays_before_block_count(tmp_path):
    # An array value must be skipped correctly so a later scalar stays aligned.
    arr = struct.pack("<I", 4) + struct.pack("<Q", 3) + struct.pack("<III", 1, 2, 3)
    data = _make_gguf(
        [
            ("general.architecture", 8, _gguf_string("qwen2")),
            ("qwen2.some_array", 9, arr),  # array of 3x uint32
            ("qwen2.block_count", 5, struct.pack("<i", 28)),  # int32
        ]
    )
    p = tmp_path / "m.gguf"
    p.write_bytes(data)
    info = read_gguf_metadata(p)
    assert info is not None and info.n_layers == 28


def test_read_gguf_metadata_bad_magic_returns_none(tmp_path):
    p = tmp_path / "notgguf.bin"
    p.write_bytes(b"NOPE" + b"\x00" * 64)
    assert read_gguf_metadata(p) is None
    assert read_gguf_metadata(tmp_path / "missing.gguf") is None


def test_read_gguf_metadata_huge_string_length_returns_none(tmp_path):
    # A malformed key with an 18-exabyte length must NOT trigger a giant read /
    # MemoryError — the reader returns None and never raises (defensive contract).
    data = (
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1)
        + struct.pack("<Q", 0xFFFFFFFFFFFFFFFF)  # key length = ~18 EB
    )
    p = tmp_path / "evil.gguf"
    p.write_bytes(data)
    assert read_gguf_metadata(p) is None


# -- WorkloadSpec / offload --------------------------------------------------


def test_offload_fraction():
    assert WorkloadSpec(n_layers=32, n_gpu_layers=16).offload_fraction == 0.5
    # -ngl 999 (or unset / negative) means "all layers".
    assert WorkloadSpec(n_layers=32, n_gpu_layers=999).offload_fraction == 1.0
    assert WorkloadSpec(n_layers=32, n_gpu_layers=None).offload_fraction == 1.0
    # Unknown layer count never falsely accuses partial offload.
    assert WorkloadSpec(n_layers=None, n_gpu_layers=10).offload_fraction == 1.0


def test_spec_roundtrip(tmp_path):
    spec = WorkloadSpec(
        model_bytes=4.5e9, n_layers=32, n_gpu_layers=32,
        mem_bandwidth_gb_s=672.0, model_name="Qwen2.5-7B-Q4", gpu_name="RTX PRO 4000",
    )
    p = tmp_path / "workload.json"
    spec.save(p)
    back = WorkloadSpec.load_or_none(p)
    assert back == spec
    assert WorkloadSpec.load_or_none(tmp_path / "none.json") is None


# -- roofline math -----------------------------------------------------------


def test_roofline_none_without_spec():
    assert roofline(None, 40.0) is None


def test_roofline_at_the_wall_full_offload():
    # 4.5 GB model fully on a 672 GB/s card -> ideal ~149 tok/s.
    spec = WorkloadSpec(model_bytes=4.5e9, n_layers=32, n_gpu_layers=32, mem_bandwidth_gb_s=672.0)
    ideal = 672e9 / 4.5e9
    r = roofline(spec, gen_tok_s=ideal * 0.8, concurrency=1.0)
    assert r is not None
    assert abs(r.ideal_tok_s - ideal) < 1e-6
    assert abs(r.mbu - 0.8) < 1e-6          # 80% of raw bandwidth
    assert r.at_bandwidth_wall is True       # >= 0.70 wall
    assert r.partial_offload is False
    # throughput% is vs the *achievable* ceiling (0.85*ideal), so ~0.94.
    assert r.throughput_pct > 0.9


def test_roofline_partial_offload_flagged():
    spec = WorkloadSpec(model_bytes=8e9, n_layers=32, n_gpu_layers=20, mem_bandwidth_gb_s=672.0)
    r = roofline(spec, gen_tok_s=10.0, concurrency=1.0)
    assert r.partial_offload is True
    assert abs(r.offload_fraction - 20 / 32) < 1e-9


def test_roofline_degrades_without_bandwidth():
    spec = WorkloadSpec(model_bytes=8e9, n_layers=32, n_gpu_layers=32)  # no bandwidth
    r = roofline(spec, gen_tok_s=10.0)
    assert r is not None
    assert r.mbu is None and r.ideal_tok_s is None
    assert r.at_bandwidth_wall is False  # never claim the wall on unknown bandwidth
