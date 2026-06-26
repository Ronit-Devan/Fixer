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
        "n",            # decline llama-tuning knobs
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


def test_setup_wizard_captures_llama_knobs(tmp_path):
    answers = iter([
        "2",                       # mode -> advise
        "",                        # no protected pids
        "",                        # no label
        "y",                       # configure llama tuning
        "/models/qwen.gguf",       # model path
        "32",                      # model layer count
        "systemctl restart llama", # restart command
    ])
    cfg = run_setup(
        input_fn=lambda prompt: next(answers, ""),
        print_fn=lambda s: None,
        path=tmp_path / "r.json",
    )
    assert cfg.knobs["model"] == "/models/qwen.gguf"
    assert cfg.knobs["model_n_layers"] == 32
    assert cfg.knobs["restart_command"] == ["systemctl", "restart", "llama"]


def test_setup_wizard_captures_draft_model_for_spec_decode(tmp_path):
    # The draft-model knob is what makes the speculative-decoding fix reachable
    # in production (the box-at-the-wall lever). Without it, that path is dead.
    answers = iter([
        "2",                  # mode -> advise
        "",                   # no protected pids
        "",                   # no label
        "y",                  # configure llama tuning
        "/models/main.gguf",  # model path
        "",                   # no layer count
        "",                   # no restart command
        "/models/draft.gguf", # draft model
        "24",                 # draft tokens per step
    ])
    cfg = run_setup(
        input_fn=lambda prompt: next(answers, ""),
        print_fn=lambda s: None,
        path=tmp_path / "r.json",
    )
    assert cfg.knobs["draft_model"] == "/models/draft.gguf"
    assert cfg.knobs["draft_n"] == 24


def test_setup_wizard_defaults_to_advise_on_empty(tmp_path):
    answers = iter(["", "", ""])  # accept defaults
    cfg = run_setup(
        input_fn=lambda prompt: next(answers, ""),
        print_fn=lambda s: None,
        path=tmp_path / "r.json",
    )
    assert cfg.mode is RemediationMode.ADVISE
