"""First-run interactive setup wizard.

The operator chooses the operating mode (the kill-switch) and the basic safety
caps the first time they run remediation, rather than inheriting a hardcoded
default. This is the text/SSH-friendly front-end; the dashboard exposes the same
settings over ``/api/remediation/config``. ``input_fn``/``print_fn`` are injected
so the wizard is unit-testable without a TTY.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from et_remediation.config import (
    RemediationConfig,
    RemediationMode,
    default_config_path,
)

_MODE_CHOICES = {
    "1": RemediationMode.OFF,
    "2": RemediationMode.ADVISE,
    "3": RemediationMode.DRY_RUN,
    "4": RemediationMode.AUTO,
}

_INTRO = """\
ET remediation setup
=====================
Choose how the remediation layer is allowed to act on this box. You can change
this any time (CLI: et-remediation mode <mode>, or the dashboard settings panel).

  1) off       do nothing (pure observability)
  2) advise    recommend fixes only; never touch the box        [safe default]
  3) dry-run   build & log the exact real commands, never run them
  4) auto      unattended auto-apply for NON-DISRUPTIVE fixes, through the
               full guarded path (apply -> verify -> confirm/rollback).
               Disruptive fixes still always require your approval.
"""


def run_setup(
    *,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    path: str | Path | None = None,
    existing: RemediationConfig | None = None,
) -> RemediationConfig:
    """Run the wizard, persist, and return the resulting config."""
    cfg = existing or RemediationConfig()
    print_fn(_INTRO)

    choice = (input_fn("Mode [2]: ") or "2").strip()
    cfg.mode = _MODE_CHOICES.get(choice, RemediationMode.ADVISE)

    pids_raw = input_fn(
        "Protected workload PID(s), comma-separated (the live task to never touch) []: "
    ).strip()
    if pids_raw:
        cfg.protected_pids = [int(p) for p in pids_raw.replace(",", " ").split() if p.strip()]
    label = input_fn("Protected workload label (e.g. pod/job name) []: ").strip()
    if label:
        cfg.protected_label = label

    if cfg.mode is RemediationMode.AUTO:
        rate = input_fn(
            f"Max auto-applies per {int(cfg.caps.window_s)}s window [{cfg.caps.max_actions_per_window}]: "
        ).strip()
        if rate:
            cfg.caps.max_actions_per_window = int(rate)
        win = input_fn(f"Verify window seconds [{int(cfg.verify_window_s)}]: ").strip()
        if win:
            cfg.verify_window_s = float(win)

    cfg.configured = True
    save_path = Path(path) if path else default_config_path()
    cfg.save(save_path)
    print_fn(f"\nSaved -> {save_path}\nMode is now: {cfg.mode.value}")
    if cfg.mode is RemediationMode.AUTO and not cfg.protected_pids:
        print_fn(
            "Note: no protected PID set. Process-touching fixes (renice / orphan-kill) "
            "will stay advise-only until a protected workload is defined."
        )
    return cfg
