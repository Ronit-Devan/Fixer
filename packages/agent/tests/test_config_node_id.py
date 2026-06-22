"""Node identity wiring: NODE_NAME -> AgentConfig.node_id, preserved on override."""

from __future__ import annotations

from gpu_doctor_agent.config import AgentConfig


def test_node_id_from_node_name_env():
    cfg = AgentConfig.from_env({"NODE_NAME": "ip-10-0-3-12"})
    assert cfg.node_id == "ip-10-0-3-12"


def test_explicit_node_id_overrides_node_name():
    cfg = AgentConfig.from_env({"NODE_NAME": "n1", "GPU_DOCTOR_NODE_ID": "explicit"})
    assert cfg.node_id == "explicit"


def test_node_id_empty_off_cluster():
    assert AgentConfig.from_env({}).node_id == ""


def test_with_overrides_preserves_node_id():
    # The CLI applies --interval via with_overrides; node identity must survive.
    cfg = AgentConfig.from_env({"NODE_NAME": "gpu-node-7"})
    overridden = cfg.with_overrides(sample_interval_s=2.0)
    assert overridden.node_id == "gpu-node-7"
    assert overridden.sample_interval_s == 2.0
