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
    # Longest matching substring wins. The SFF key ("pro 4000 blackwell sff")
    # must beat the full-size key ("pro 4000 blackwell") which in turn beats the
    # generic "blackwell" fallback — the SFF is 432 GB/s (18 Gbps), NOT the
    # full-size 672 (28 Gbps). (This assertion previously codified the bug.)
    assert bandwidth_for("NVIDIA RTX PRO 4000 Blackwell SFF Edition") == 432.0
    assert bandwidth_for("NVIDIA RTX PRO 4000 Blackwell") == 672.0
    assert bandwidth_for("NVIDIA RTX PRO 9999 Blackwell") == 672.0  # generic fallback
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


# -- MoE active-byte accounting (decode ceiling ~10x off if full weights used) --


def _tensor_info(name: str, size: int, offset: int) -> bytes:
    # name, n_dims=1, dims=[max(size,1)], ggml_type=0 (f32), offset
    return (
        _gguf_string(name)
        + struct.pack("<I", 1)
        + struct.pack("<Q", max(size, 1))
        + struct.pack("<I", 0)
        + struct.pack("<Q", offset)
    )


def _make_gguf_with_tensors(
    kvs: list[tuple[str, int, bytes]], tensors: list[tuple[str, int]]
) -> bytes:
    # tensors: [(name, size_bytes)] laid out contiguously in the data section.
    offsets, acc = [], 0
    for _, sz in tensors:
        offsets.append(acc)
        acc += sz
    total_data = acc
    blob = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", len(tensors))
        + struct.pack("<Q", len(kvs))
    )
    for key, vtype, val in kvs:
        blob += _gguf_string(key) + struct.pack("<I", vtype) + val
    for (name, sz), off in zip(tensors, offsets):
        blob += _tensor_info(name, sz, off)
    blob += b"\x00" * ((-len(blob)) % 32)  # align data section to 32
    blob += b"\x00" * total_data
    return blob


def test_moe_active_bytes_scales_by_used_over_count(tmp_path):
    # 128 experts, 8/token. Non-expert=2GB, routed experts=8GB, total=10GB.
    # active = 10 - 8*(1 - 8/128) = 10 - 8*0.9375 = 2.5 GB.
    kvs = [
        ("general.architecture", 8, _gguf_string("qwen3moe")),
        ("qwen3moe.block_count", 4, struct.pack("<I", 48)),
        ("qwen3moe.expert_count", 4, struct.pack("<I", 128)),
        ("qwen3moe.expert_used_count", 4, struct.pack("<I", 8)),
    ]
    tensors = [
        ("token_embd.weight", 1_000_000_000),
        ("blk.0.attn_q.weight", 500_000_000),
        ("blk.0.ffn_gate_inp.weight", 100_000_000),  # router, always active
        ("blk.0.ffn_gate_shexp.weight", 400_000_000),  # shared, always active
        ("blk.0.ffn_gate_exps.weight", 3_000_000_000),  # routed
        ("blk.0.ffn_up_exps.weight", 3_000_000_000),  # routed
        ("blk.0.ffn_down_exps.weight", 2_000_000_000),  # routed
    ]
    p = tmp_path / "moe.gguf"
    p.write_bytes(_make_gguf_with_tensors(kvs, tensors))
    info = read_gguf_metadata(p)
    assert info is not None
    assert info.expert_count == 128 and info.expert_used_count == 8
    assert info.is_moe is True
    assert info.n_layers == 48
    assert info.active_bytes is not None
    assert abs(info.active_bytes - 2.5e9) < 1e6  # ~2.5 GB, not the 10 GB file


def test_dense_model_has_no_active_bytes(tmp_path):
    # No expert keys -> dense -> active_bytes stays None (decode uses full weights).
    kvs = [
        ("general.architecture", 8, _gguf_string("llama")),
        ("llama.block_count", 4, struct.pack("<I", 32)),
    ]
    tensors = [("token_embd.weight", 1_000_000_000), ("blk.0.ffn_gate.weight", 2_000_000_000)]
    p = tmp_path / "dense.gguf"
    p.write_bytes(_make_gguf_with_tensors(kvs, tensors))
    info = read_gguf_metadata(p)
    assert info is not None
    assert info.n_layers == 32
    assert info.is_moe is False
    assert info.active_bytes is None


def test_moe_missing_expert_tensors_degrades_to_none(tmp_path):
    # Expert COUNT keys present but no *_exps tensors -> can't size it -> None,
    # while expert_count/used are still surfaced (graceful degradation).
    kvs = [
        ("general.architecture", 8, _gguf_string("qwen3moe")),
        ("qwen3moe.block_count", 4, struct.pack("<I", 4)),
        ("qwen3moe.expert_count", 4, struct.pack("<I", 60)),
        ("qwen3moe.expert_used_count", 4, struct.pack("<I", 4)),
    ]
    tensors = [("token_embd.weight", 1_000_000_000), ("blk.0.attn_q.weight", 500_000_000)]
    p = tmp_path / "moe_bad.gguf"
    p.write_bytes(_make_gguf_with_tensors(kvs, tensors))
    info = read_gguf_metadata(p)
    assert info is not None
    assert info.expert_count == 60 and info.expert_used_count == 4
    assert info.active_bytes is None  # no *_exps -> couldn't size, fell back


def test_workloadspec_bytes_on_gpu_prefers_active_bytes():
    # MoE spec: decode reads active_bytes, not the full file.
    moe = WorkloadSpec(model_bytes=20e9, active_bytes=2e9, n_layers=48, n_gpu_layers=999)
    assert moe.is_moe is True
    assert moe.bytes_on_gpu() == 2e9  # full offload -> active bytes exactly
    # Dense spec: falls back to full model bytes.
    dense = WorkloadSpec(model_bytes=8e9, n_layers=32, n_gpu_layers=999)
    assert dense.is_moe is False
    assert dense.bytes_on_gpu() == 8e9


def test_roofline_moe_ceiling_uses_active_bytes():
    from et_monitor.perf import roofline

    # 432 GB/s SFF, MoE reading ~4 GB/token -> ideal ~108 tok/s (not ~18 for a
    # 20 GB dense-style denominator). A measured 90 tok/s is a healthy ~0.83 MBU.
    spec = WorkloadSpec(
        model_bytes=20e9, active_bytes=4e9, n_layers=48, n_gpu_layers=999,
        mem_bandwidth_gb_s=432.0,
    )
    rl = roofline(spec, gen_tok_s=90.0)
    assert rl is not None and rl.is_moe is True
    assert 100 < rl.ideal_tok_s < 115  # 432e9/4e9 = 108
    assert rl.mbu is not None and 0.7 < rl.mbu < 0.95  # healthy, not absurd >1


# -- KV-cache budget from GGUF (weights + KV must fit 24 GB) ------------------


def test_kv_bytes_per_token_from_gguf_attention_keys(tmp_path):
    # 48 layers, 8 KV heads (GQA), n_embd 4096, 32 query heads -> head_dim 128.
    # kv/token = 2 * 48 * 8 * 128 * 2 bytes = 196608 bytes.
    kvs = [
        ("general.architecture", 8, _gguf_string("qwen3moe")),
        ("qwen3moe.block_count", 4, struct.pack("<I", 48)),
        ("qwen3moe.attention.head_count_kv", 4, struct.pack("<I", 8)),
        ("qwen3moe.attention.head_count", 4, struct.pack("<I", 32)),
        ("qwen3moe.embedding_length", 4, struct.pack("<I", 4096)),
        ("qwen3moe.context_length", 4, struct.pack("<I", 32768)),
    ]
    p = tmp_path / "kv.gguf"
    p.write_bytes(_make_gguf_with_tensors(kvs, [("token_embd.weight", 1000)]))
    info = read_gguf_metadata(p)
    assert info is not None
    assert info.n_head_kv == 8 and info.n_head == 32 and info.n_embd == 4096
    assert info.context_length == 32768
    assert info.kv_bytes_per_token() == 2 * 48 * 8 * 128 * 2  # 196608


def test_kv_bytes_none_when_attention_keys_missing(tmp_path):
    kvs = [
        ("general.architecture", 8, _gguf_string("llama")),
        ("llama.block_count", 4, struct.pack("<I", 32)),
    ]
    p = tmp_path / "nokv.gguf"
    p.write_bytes(_make_gguf_with_tensors(kvs, [("token_embd.weight", 1000)]))
    info = read_gguf_metadata(p)
    assert info is not None and info.kv_bytes_per_token() is None


def test_vram_fit_budget_math():
    # Client shape: ~18 GB weights, ~196 KB/token KV, 24 GB card.
    spec = WorkloadSpec(
        model_bytes=18e9, kv_bytes_per_token=196608.0, mem_total_bytes=24e9,
        n_layers=48, n_gpu_layers=999, mem_bandwidth_gb_s=432.0,
    )
    fit = spec.vram_fit(30000)
    assert fit is not None
    # KV @ 30K = 196608 * 30000 ≈ 5.9 GB; weights 18 + 5.9 + 0.6 ≈ 24.5 GB > 24 -> tight/overflow
    assert fit["kv_gb"] > 5.0
    assert fit["fits"] is False  # 30K context does NOT fit alongside 18 GB weights
    # A shorter context fits with headroom.
    assert spec.vram_fit(8000)["fits"] is True


def test_vram_fit_none_without_facts():
    assert WorkloadSpec(model_bytes=18e9).vram_fit(30000) is None  # no kv/mem facts
