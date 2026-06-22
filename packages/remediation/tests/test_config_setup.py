"""Config persistence + the first-run setup wizard."""

from __future__ import annotations

from et_remediation.config import (
    CapsConfig,
    RemediationConfig,
    RemediationMode,
)
from et_remediation.setup_ui import run_setup


def test_config_roundtrip(tmp_path):
    cfg = RemediationConfig(
        mode=RemediationMode.AUTO,
        verify_window_s=20.0,
        caps=CapsConfig(max_actions_per_window=5),
        protected_pids=[1, 2, 3],
        configured=True,
    )
    p = tmp_path / "remediation.json"
    cfg.save(p)
    back = RemediationConfig.load(p)
    assert back.mode is RemediationMode.AUTO
    assert back.verify_window_s == 20.0
    assert back.caps.max_actions_per_window == 5
    assert back.protected_pids == [1, 2, 3]
    assert back.configured is True


def test_load_or_default_when_missing(tmp_path):
    cfg = RemediationConfig.load_or_default(tmp_path / "nope.json")
    assert cfg.mode is RemediationMode.ADVISE  # safe default
    assert cfg.configured is False


def test_load_or_default_on_corrupt_file_fails_safe(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json", encoding="utf-8")
    cfg = RemediationConfig.load_or_default(p)
    assert cfg.mode is RemediationMode.ADVISE


def test_mode_kill_switch_properties():
    assert RemediationMode.AUTO.actuates is True
    for m in (RemediationMode.OFF, RemediationMode.ADVISE, RemediationMode.DRY_RUN):
        assert m.actuates is False
    assert RemediationMode.OFF.considers_actions is False
    assert RemediationMode.ADVISE.considers_actions is True


def test_setup_wizard_writes_config(tmp_path):
    answers = iter([
        "4",            # mode -> auto
        "111, 222",     # protected pids
        "live-job",     # label
        "5",            # max actions/window
        "15",           # verify window
    ])
    out_lines = []
    p = tmp_path / "remediation.json"
    cfg = run_setup(
        input_fn=lambda prompt: next(answers),
        print_fn=out_lines.append,
        path=p,
    )
    assert cfg.mode is RemediationMode.AUTO
    assert cfg.protected_pids == [111, 222]
    assert cfg.protected_label == "live-job"
    assert cfg.caps.max_actions_per_window == 5
    assert cfg.verify_window_s == 15.0
    assert cfg.configured is True
    # Persisted and reloadable.
    assert RemediationConfig.load(p).mode is RemediationMode.AUTO


def test_setup_wizard_defaults_to_advise_on_empty(tmp_path):
    answers = iter(["", "", ""])  # accept defaults
    cfg = run_setup(
        input_fn=lambda prompt: next(answers, ""),
        print_fn=lambda s: None,
        path=tmp_path / "r.json",
    )
    assert cfg.mode is RemediationMode.ADVISE
