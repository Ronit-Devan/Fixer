"""The snapshot() 'workload' block the dashboard reads for the ceiling panel."""

from __future__ import annotations

from et_monitor.gpu import MockGpuSampler
from et_monitor.perf import WorkloadSpec
from et_monitor.state import Monitor, MonitorConfig, _workload_dict


def test_workload_dict_none_without_spec():
    assert _workload_dict(None) is None


def test_workload_dict_carries_ceiling_and_offload():
    spec = WorkloadSpec(
        model_bytes=4.5e9, n_layers=32, n_gpu_layers=16, mem_bandwidth_gb_s=672.0,
        model_name="Qwen2.5-7B-Q4", gpu_name="RTX PRO 4000 Blackwell",
    )
    d = _workload_dict(spec)
    assert d["ceiling_tok_s"] is not None and d["ceiling_tok_s"] > 0
    assert d["offload_fraction"] == 0.5
    assert d["model_gb"] == 4.5


def test_snapshot_includes_workload_block():
    spec = WorkloadSpec(model_bytes=4.5e9, n_layers=32, n_gpu_layers=32, mem_bandwidth_gb_s=672.0)
    mon = Monitor(MockGpuSampler(), None, MonitorConfig(workload_spec=spec))
    mon.tick()
    snap = mon.snapshot()
    assert "workload" in snap and snap["workload"]["ceiling_tok_s"] is not None


def test_snapshot_workload_none_when_unconfigured():
    mon = Monitor(MockGpuSampler(), None, MonitorConfig())
    mon.tick()
    assert mon.snapshot()["workload"] is None
