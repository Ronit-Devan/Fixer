"""Single-stream decode roofline: turn raw counters into "how close to the
physical ceiling are we", so a bare number like "40%" stops being ambiguous.

A llama.cpp box serving ONE stream is *memory-bandwidth bound* during decode:
every generated token streams the (quantized) weights resident on the GPU once.
So the single-stream decode ceiling is

    ideal_tok_s  = mem_bandwidth_bytes_s / model_bytes_on_gpu
    ceiling_tok_s = achievable_fraction * ideal_tok_s     (real kernels reach
                                                           ~70-90% of spec BW)
    MBU          = gen_tok_s / ideal_tok_s                (fraction of raw BW)
    throughput%  = gen_tok_s / ceiling_tok_s              (fraction of the
                                                           *achievable* peak)

Reading these straight resolves the whole "40% of what?" question:

  * MBU near the wall (>= ~0.7) on a fully-offloaded model -> you are at the
    physical single-stream limit. Raising GPU *utilization* to 90% is
    impossible at concurrency 1; the only levers are batching (more concurrent
    requests), speculative decoding, or a smaller/faster quant.
  * Low MBU while a fully-offloaded model is actively decoding -> real, fixable
    waste (throttling, a host/sampling bottleneck, a tiny batch).
  * Layers on CPU (``n_gpu_layers < n_layers``) -> partial offload: the single
    biggest, most common llama.cpp throughput bug, and a precise fix (-ngl).

Everything here is pure and O(1) on the hot path: the hardware/model facts come
from a ``WorkloadSpec`` captured ONCE at setup (GGUF size + layer count + the
card's memory bandwidth), never probed per tick. The GGUF reader and bandwidth
lookup run only at setup/startup.
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

# --- GPU memory bandwidth (GB/s) -------------------------------------------
# Substring-keyed table of *spec* HBM/GDDR bandwidth for the cards we see most.
# These are starting estimates the setup wizard lets the operator override; the
# methodology (gen_tok_s vs bandwidth ceiling) is what matters, not the third
# significant figure. Matching is longest-substring-wins on a lowercased name.
_BANDWIDTH_GB_S: dict[str, float] = {
    "h200": 4800.0,
    "h100": 3350.0,
    "a100": 2039.0,  # 80GB SXM; 40GB PCIe is ~1555, override at setup
    "a40": 696.0,
    "a10": 600.0,
    "l40s": 864.0,
    "l40": 864.0,
    "l4": 300.0,
    "v100": 900.0,
    "t4": 320.0,
    # RTX / consumer + workstation
    "5090": 1792.0,
    "4090": 1008.0,
    "4080": 717.0,
    "4070": 504.0,
    "3090": 936.0,
    "3080": 760.0,
    "a6000": 768.0,
    "6000 ada": 960.0,
    "rtx 6000 ada": 960.0,
    "5000 ada": 576.0,
    "4500 ada": 432.0,
    "4000 ada": 360.0,
    "2000 ada": 288.0,
    # Blackwell workstation (RTX PRO ... Blackwell) — approx, refine at setup
    "rtx pro 6000 blackwell": 1792.0,
    "pro 5000 blackwell": 1344.0,
    "pro 4500 blackwell": 896.0,
    "pro 4000 blackwell": 672.0,
    "blackwell": 672.0,  # generic fallback for an unrecognized Blackwell SKU
}


def bandwidth_for(gpu_name: str | None) -> float | None:
    """Best-effort spec memory bandwidth (GB/s) for a GPU name, else None.

    Longest matching substring wins so "RTX PRO 4000 Blackwell" prefers the
    specific entry over the generic "blackwell" fallback.
    """
    if not gpu_name:
        return None
    name = gpu_name.lower()
    best: tuple[int, float] | None = None
    for key, gbs in _BANDWIDTH_GB_S.items():
        if key in name and (best is None or len(key) > best[0]):
            best = (len(key), gbs)
    return best[1] if best else None


# --- GGUF metadata (read ONCE at setup) ------------------------------------

_GGUF_MAGIC = b"GGUF"
# GGUF value type tag -> (struct format, byte size) for the fixed-width scalars.
_GGUF_FIXED: dict[int, tuple[str, int]] = {
    0: ("<B", 1),  # uint8
    1: ("<b", 1),  # int8
    2: ("<H", 2),  # uint16
    3: ("<h", 2),  # int16
    4: ("<I", 4),  # uint32
    5: ("<i", 4),  # int32
    6: ("<f", 4),  # float32
    7: ("<?", 1),  # bool
    10: ("<Q", 8),  # uint64
    11: ("<q", 8),  # int64
    12: ("<d", 8),  # float64
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9
# A sane upper bound on a single GGUF string (keys / names / arch are tiny). A
# malformed length field (e.g. 0xFFFF...) must NOT trigger an 18-exabyte read —
# reject it so the reader keeps its "never raises, returns None" contract.
_MAX_GGUF_STR = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class GgufInfo:
    """The handful of GGUF facts the roofline needs."""

    path: str
    file_bytes: int
    architecture: str | None = None
    name: str | None = None
    n_layers: int | None = None  # <arch>.block_count
    param_count: int | None = None  # general.parameter_count, if present


def _read_gguf_string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    if n > _MAX_GGUF_STR:
        raise ValueError(f"gguf string length {n} implausible; refusing to read")
    return f.read(n).decode("utf-8", "replace")


def _skip_gguf_value(f, vtype: int, depth: int = 0) -> None:
    """Advance the file cursor past one GGUF value of ``vtype`` without keeping
    it. Strings/arrays are skipped by length so we never read vocab arrays."""
    if vtype in _GGUF_FIXED:
        f.seek(_GGUF_FIXED[vtype][1], 1)
    elif vtype == _GGUF_STRING:
        (n,) = struct.unpack("<Q", f.read(8))
        f.seek(n, 1)
    elif vtype == _GGUF_ARRAY:
        if depth > 4:
            raise ValueError("gguf array nested too deep")
        (elem_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if elem_type in _GGUF_FIXED:
            f.seek(_GGUF_FIXED[elem_type][1] * count, 1)
        else:
            for _ in range(count):
                _skip_gguf_value(f, elem_type, depth + 1)
    else:
        raise ValueError(f"unknown gguf value type {vtype}")


def read_gguf_metadata(path: str | Path, *, max_kv: int = 4000) -> GgufInfo | None:
    """Read the few GGUF header facts we need (layer count, arch, size).

    Defensive by construction: any malformed/unsupported header returns ``None``
    rather than raising, and the scan early-exits the moment it has both the
    architecture and the layer count, so the giant tokenizer arrays are never
    touched. Runs at setup only.
    """
    p = Path(path)
    try:
        file_bytes = p.stat().st_size
        with p.open("rb") as f:
            if f.read(4) != _GGUF_MAGIC:
                return None
            (version,) = struct.unpack("<I", f.read(4))
            if version not in (2, 3):
                # v1 used 32-bit counts; we don't bother — return size-only info.
                return GgufInfo(path=str(p), file_bytes=file_bytes)
            (_tensor_count,) = struct.unpack("<Q", f.read(8))
            (kv_count,) = struct.unpack("<Q", f.read(8))

            arch: str | None = None
            name: str | None = None
            n_layers: int | None = None
            param_count: int | None = None

            for _ in range(min(kv_count, max_kv)):
                key = _read_gguf_string(f)
                (vtype,) = struct.unpack("<I", f.read(4))
                if vtype in _GGUF_FIXED and (
                    key.endswith(".block_count")
                    or key == "general.parameter_count"
                    or key == "general.architecture"  # never fixed, but be safe
                ):
                    raw = f.read(_GGUF_FIXED[vtype][1])
                    (val,) = struct.unpack(_GGUF_FIXED[vtype][0], raw)
                    if key.endswith(".block_count"):
                        n_layers = int(val)
                    elif key == "general.parameter_count":
                        param_count = int(val)
                elif vtype == _GGUF_STRING and key in (
                    "general.architecture",
                    "general.name",
                ):
                    s = _read_gguf_string(f)
                    if key == "general.architecture":
                        arch = s
                    else:
                        name = s
                else:
                    _skip_gguf_value(f, vtype)
                if arch is not None and n_layers is not None:
                    break  # have what we need; skip the tokenizer arrays entirely
            return GgufInfo(
                path=str(p),
                file_bytes=file_bytes,
                architecture=arch,
                name=name,
                n_layers=n_layers,
                param_count=param_count,
            )
    except (OSError, struct.error, ValueError, MemoryError, OverflowError):
        return None


# --- the workload spec + roofline ------------------------------------------


@dataclass(frozen=True)
class WorkloadSpec:
    """Static hardware+model facts the roofline needs. Captured once at setup.

    Any field may be None — the roofline degrades gracefully (returns Nones for
    whatever it can't compute) so the analyzer simply falls back to its
    util-based heuristics when the spec is absent or partial.
    """

    model_bytes: float | None = None  # full quantized weight bytes (~GGUF size)
    n_layers: int | None = None
    n_gpu_layers: int | None = None  # configured -ngl; None/<0 => treat as all
    mem_bandwidth_gb_s: float | None = None
    achievable_fraction: float = 0.85  # realistic share of spec BW real kernels hit
    wall_mbu: float = 0.70  # MBU at/above this = "at the single-stream wall"
    # Display-only context.
    model_name: str | None = None
    gpu_name: str | None = None

    @property
    def offload_fraction(self) -> float:
        """Fraction of the model's layers actually resident on the GPU (0..1).

        Unknown layer count or unset -ngl is treated as fully offloaded (1.0) so
        we never cry "partial offload" on missing data. ``-ngl`` >= n_layers (or
        the conventional 999) clamps to 1.0.
        """
        if not self.n_layers or self.n_layers <= 0:
            return 1.0
        if self.n_gpu_layers is None or self.n_gpu_layers < 0:
            return 1.0
        return max(0.0, min(1.0, self.n_gpu_layers / self.n_layers))

    @property
    def has_roofline(self) -> bool:
        return bool(self.model_bytes and self.mem_bandwidth_gb_s)

    def bytes_on_gpu(self) -> float | None:
        if self.model_bytes is None:
            return None
        return self.model_bytes * self.offload_fraction

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkloadSpec":
        fields = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**fields)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_or_none(cls, path: str | Path | None) -> "WorkloadSpec | None":
        if path is None:
            return None
        p = Path(path)
        if not p.is_file():
            return None
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class Roofline:
    """The decode roofline evaluated against a live generation rate."""

    gen_tok_s: float | None
    ideal_tok_s: float | None  # bandwidth / full-model bytes (100% BW, full offload)
    ceiling_tok_s: float | None  # achievable_fraction * ideal_tok_s
    mbu: float | None  # gen / ideal  (fraction of raw bandwidth used)
    throughput_pct: float | None  # gen / ceiling (fraction of achievable peak)
    offload_fraction: float
    concurrency: float | None
    partial_offload: bool
    at_bandwidth_wall: bool

    def to_metrics(self) -> dict:
        """Flatten into the diagnosis ``metrics`` dict (rounded, JSON-friendly)."""

        def r(v, n=2):
            return round(v, n) if isinstance(v, (int, float)) else v

        return {
            "gen_tokens_per_s": r(self.gen_tok_s, 1),
            "ideal_tok_s": r(self.ideal_tok_s, 1),
            "ceiling_tok_s": r(self.ceiling_tok_s, 1),
            "mbu": r(self.mbu, 3),
            "throughput_pct": r(self.throughput_pct, 3),
            "offload_fraction": r(self.offload_fraction, 3),
            "concurrency": r(self.concurrency, 2),
            "partial_offload": self.partial_offload,
            "at_bandwidth_wall": self.at_bandwidth_wall,
        }


def roofline(
    spec: WorkloadSpec | None,
    gen_tok_s: float | None,
    *,
    concurrency: float | None = None,
) -> Roofline | None:
    """Evaluate the decode roofline. None if there's no spec to reason with.

    The ceiling is single-stream (concurrency 1): each token streams the
    GPU-resident weights once. ``mbu`` can exceed 1.0 when continuous batching
    amortizes the weight read across several concurrent tokens — which is itself
    the signal that batching is already lifting you past the single-stream wall.
    """
    if spec is None:
        return None
    offload = spec.offload_fraction
    partial = offload < 0.98
    bytes_on_gpu = spec.bytes_on_gpu()
    bw_bytes_s = (
        spec.mem_bandwidth_gb_s * 1e9 if spec.mem_bandwidth_gb_s else None
    )
    ideal = ceiling = mbu = tput = None
    if bw_bytes_s and bytes_on_gpu and bytes_on_gpu > 0:
        ideal = bw_bytes_s / bytes_on_gpu
        ceiling = spec.achievable_fraction * ideal
        if gen_tok_s is not None and ideal > 0:
            mbu = gen_tok_s / ideal
            tput = gen_tok_s / ceiling if ceiling > 0 else None
    at_wall = bool(mbu is not None and mbu >= spec.wall_mbu)
    return Roofline(
        gen_tok_s=gen_tok_s,
        ideal_tok_s=ideal,
        ceiling_tok_s=ceiling,
        mbu=mbu,
        throughput_pct=tput,
        offload_fraction=offload,
        concurrency=concurrency,
        partial_offload=partial,
        at_bandwidth_wall=at_wall,
    )


def default_spec_path() -> Path:
    """Canonical location for the persisted workload spec (~/.et/workload.json)."""
    return Path.home() / ".et" / "workload.json"
