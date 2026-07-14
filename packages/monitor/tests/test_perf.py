"""Tests for the decode roofline + GGUF metadata reader (perf.py)."""

from __future__ import annotations

import struct

from et_monitor.perf import (
    GgufInfo,
    WorkloadSpec,
    bandwidth_for,
    estimate_moe_active_bytes,
    read_gguf_metadata,
    roofline,
)


# -- bandwidth lookup --------------------------------------------------------


def test_bandwidth_longest_substring_wins():
    assert bandwidth_for("Tesla H100 PCIe") == 3350.0
    assert bandwidth_for("Some Unknown Card 9000") is None
    assert bandwidth_for(None) is None


def test_bandwidth_sff_disambiguated_from_full_blackwell():
    # The SFF trap: the 70 W SFF downclocks its GDDR7 to 432 GB/s, while the full
    # 140 W card is 672. The SFF NVML name contains "pro 4000 blackwell", so the
    # longer "pro 4000 blackwell sff" key MUST win — otherwise the SFF silently
    # inherits the full card's 672 and the roofline ceiling is ~1.55x too high.
    assert bandwidth_for("NVIDIA RTX PRO 4000 Blackwell SFF Edition") == 432.0
    assert bandwidth_for("NVIDIA RTX PRO 4000 Blackwell SFF") == 432.0
    # The full (non-SFF) card is unaffected and stays 672.
    assert bandwidth_for("NVIDIA RTX PRO 4000 Blackwell") == 672.0


def test_bandwidth_blackwell_workstation_lineup():
    # Every workstation Blackwell SKU resolves to its own verified bandwidth
    # rather than the generic "blackwell" fallback.
    assert bandwidth_for("NVIDIA RTX PRO 6000 Blackwell Workstation Edition") == 1792.0
    assert bandwidth_for("NVIDIA RTX PRO 5000 Blackwell") == 1344.0
    assert bandwidth_for("NVIDIA RTX PRO 4500 Blackwell") == 896.0
    assert bandwidth_for("NVIDIA RTX PRO 2000 Blackwell") == 288.0
    # An unrecognized Blackwell SKU still gets the generic fallback (not None).
    assert bandwidth_for("NVIDIA RTX PRO 9999 Blackwell") == 672.0


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


# -- MoE active-bytes estimate ----------------------------------------------


def _moe_info(**kw) -> GgufInfo:
    """A Qwen3-30B-A3B-shaped MoE GgufInfo (~3B of 30B active)."""
    base = dict(
        path="m.gguf",
        file_bytes=17_000_000_000,  # ~17 GB Q4 MoE footprint
        architecture="qwen3moe",
        n_layers=48,
        param_count=30_500_000_000,
        expert_count=128,
        expert_used_count=8,
        embedding_length=2048,
        expert_ffn_len=768,
    )
    base.update(kw)
    return GgufInfo(**base)


def test_estimate_active_bytes_dense_returns_none():
    info = GgufInfo(path="m", file_bytes=16_000_000_000, architecture="llama", n_layers=32)
    ab, note = estimate_moe_active_bytes(info)
    assert ab is None
    assert "dense" in note.lower()
    # expert_count of 1 is not a real MoE.
    ab2, _ = estimate_moe_active_bytes(_moe_info(expert_count=1))
    assert ab2 is None


def test_estimate_active_bytes_moe_geometry():
    # ~3B of 30B active -> ~10-11% of weights streamed, NOT the full 30B.
    ab, note = estimate_moe_active_bytes(_moe_info())
    assert ab is not None
    frac = ab / 17_000_000_000
    assert 0.09 < frac < 0.13
    assert ab < 17_000_000_000
    assert note.startswith("MoE")
    assert "%" in note


def test_estimate_active_bytes_coarse_fallback():
    # Only expert counts known (no d_model / ffn / param_count) -> routed ratio.
    info = _moe_info(embedding_length=None, expert_ffn_len=None, param_count=None)
    ab, note = estimate_moe_active_bytes(info)
    assert ab is not None
    assert abs(ab - 17_000_000_000 * (8 / 128)) < 1.0
    assert "coarse" in note.lower()


def test_estimate_active_bytes_missing_used_count_is_dense():
    # expert_count without expert_used_count can't be reasoned about -> dense.
    ab, _ = estimate_moe_active_bytes(_moe_info(expert_used_count=None))
    assert ab is None


def test_estimate_active_frac_clamped_to_expert_ratio():
    # A garbage (too-small) param_count must not push the active fraction below
    # the raw routed-expert ratio.
    ab, _ = estimate_moe_active_bytes(_moe_info(param_count=1_000_000))
    assert ab is not None
    assert ab >= 17_000_000_000 * (8 / 128) - 1.0


def test_read_gguf_metadata_reads_moe_geometry_and_stops_at_tokenizer(tmp_path):
    U32, U64 = 4, 10
    data = _make_gguf(
        [
            ("general.architecture", 8, _gguf_string("qwen3moe")),
            ("qwen3moe.block_count", U32, struct.pack("<I", 48)),
            ("qwen3moe.embedding_length", U32, struct.pack("<I", 2048)),
            ("qwen3moe.expert_count", U32, struct.pack("<I", 128)),
            ("qwen3moe.expert_used_count", U32, struct.pack("<I", 8)),
            ("qwen3moe.expert_feed_forward_length", U32, struct.pack("<I", 768)),
            ("general.parameter_count", U64, struct.pack("<Q", 30_500_000_000)),
            # Tokenizer arrays come last; the reader must STOP at this key and
            # never parse the (deliberately invalid) array value that follows.
            ("tokenizer.ggml.tokens", 9, b"\xff\xff"),
        ]
    )
    p = tmp_path / "moe.gguf"
    p.write_bytes(data)
    info = read_gguf_metadata(p)
    assert info is not None
    assert info.architecture == "qwen3moe"
    assert info.n_layers == 48
    assert info.expert_count == 128
    assert info.expert_used_count == 8
    assert info.embedding_length == 2048
    assert info.expert_ffn_len == 768
    assert info.param_count == 30_500_000_000
    assert info.is_moe is True
    ab, _ = estimate_moe_active_bytes(info)
    assert ab is not None and ab < info.file_bytes


def test_per_token_bytes_dense_vs_moe():
    dense = WorkloadSpec(model_bytes=16e9, n_layers=48, n_gpu_layers=48)
    assert dense.per_token_bytes() == 16e9
    assert dense.is_moe is False
    moe = WorkloadSpec(model_bytes=16e9, active_bytes=1.8e9, n_layers=48, n_gpu_layers=48)
    assert moe.per_token_bytes() == 1.8e9
    assert moe.is_moe is True
    # Partial offload scales the streamed bytes.
    half = WorkloadSpec(model_bytes=16e9, active_bytes=1.8e9, n_layers=48, n_gpu_layers=24)
    assert abs(half.per_token_bytes() - 1.8e9 * 0.5) < 1.0


def test_roofline_uses_active_bytes_for_moe():
    # This is the client case. Same 16 GB footprint + 432 GB/s SFF: the MoE
    # ceiling reflects ACTIVE bytes, ~10x higher than the (wrong) dense reading.
    bw = 432.0
    dense = WorkloadSpec(model_bytes=16e9, n_layers=48, n_gpu_layers=48, mem_bandwidth_gb_s=bw)
    moe = WorkloadSpec(
        model_bytes=16e9, active_bytes=1.8e9, n_layers=48, n_gpu_layers=48,
        mem_bandwidth_gb_s=bw,
    )
    rd = roofline(dense, gen_tok_s=None)
    rm = roofline(moe, gen_tok_s=None)
    assert abs(rd.ideal_tok_s - 432e9 / 16e9) < 1e-6      # ~27 tok/s
    assert abs(rm.ideal_tok_s - 432e9 / 1.8e9) < 1e-6     # ~240 tok/s
    assert rm.ideal_tok_s > 8 * rd.ideal_tok_s
    # A measured 94 tok/s is a healthy ~0.39 MBU on the MoE, not the impossible
    # >3x-over-ceiling that the dense assumption would (wrongly) report.
    assert 0.3 < roofline(moe, gen_tok_s=94.0).mbu < 0.5
    assert roofline(dense, gen_tok_s=94.0).mbu > 3.0
