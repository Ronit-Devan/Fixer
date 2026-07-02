"""First-run interactive setup wizard.

The operator chooses the operating mode (the kill-switch) and the basic safety
caps the first time they run remediation, rather than inheriting a hardcoded
default. This is the text/SSH-friendly front-end; the dashboard exposes the same
settings over ``/api/remediation/config``. ``input_fn``/``print_fn`` are injected
so the wizard is unit-testable without a TTY.
"""

from __future__ import annotations

import shlex
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

    # A headless/piped invocation (systemd, `< /dev/null`, closed-stdin SSH)
    # raises EOFError from input(); every prompt treats that as "accept the
    # default" so setup completes with the safe ADVISE config instead of dying
    # with a traceback and never saving anything.
    eof_seen = False

    def ask(prompt: str) -> str:
        nonlocal eof_seen
        try:
            return input_fn(prompt)
        except EOFError:
            if not eof_seen:
                eof_seen = True
                print_fn("(no interactive input available; accepting defaults)")
            return ""

    choice = (ask("Mode [2]: ") or "2").strip()
    cfg.mode = _MODE_CHOICES.get(choice, RemediationMode.ADVISE)

    pids_raw = ask(
        "Protected workload PID(s), comma-separated (the live task to never touch) []: "
    ).strip()
    if pids_raw:
        try:
            cfg.protected_pids = [
                int(p) for p in pids_raw.replace(",", " ").split() if p.strip()
            ]
        except ValueError:
            print_fn(f"  (couldn't parse {pids_raw!r} as PIDs; protected PIDs left unset)")
    label = ask("Protected workload label (e.g. pod/job name) []: ").strip()
    if label:
        cfg.protected_label = label

    if cfg.mode is RemediationMode.AUTO:
        rate = ask(
            f"Max auto-applies per {int(cfg.caps.window_s)}s window [{cfg.caps.max_actions_per_window}]: "
        ).strip()
        if rate:
            try:
                cfg.caps.max_actions_per_window = int(rate)
            except ValueError:
                print_fn(f"  (not a number: {rate!r}; keeping {cfg.caps.max_actions_per_window})")
        win = ask(f"Verify window seconds [{int(cfg.verify_window_s)}]: ").strip()
        if win:
            try:
                cfg.verify_window_s = float(win)
            except ValueError:
                print_fn(f"  (not a number: {win!r}; keeping {int(cfg.verify_window_s)})")

    # llama.cpp tuning knobs: what a RESTART_LLAMA_SERVER fix needs to actually
    # relaunch the server (raise -ngl, add slots). Optional — the strategies have
    # safe defaults (-ngl 999, demand-gated --parallel) when these are absent.
    want_llama = ask(
        "Configure llama.cpp tuning for auto-fixes (model / layers / restart cmd)? [y/N]: "
    ).strip().lower()
    if want_llama.startswith("y"):
        model = ask("  Model .gguf path []: ").strip()
        if model:
            cfg.knobs["model"] = model
        layers = ask(
            "  Model layer count (enables '-ngl all' on partial offload) []: "
        ).strip()
        if layers:
            try:
                cfg.knobs["model_n_layers"] = int(layers)
            except ValueError:
                pass
        restart = ask("  Command to relaunch llama-server []: ").strip()
        if restart:
            # Quote-aware split so a model path with spaces survives ("-m
            # '/models/my model.gguf'"). Unquoted commands keep the plain
            # whitespace split (identical to prior behaviour, and safe for
            # Windows backslash paths, which shlex would mangle).
            if '"' in restart or "'" in restart:
                try:
                    cfg.knobs["restart_command"] = shlex.split(restart)
                except ValueError:
                    cfg.knobs["restart_command"] = restart.split()
            else:
                cfg.knobs["restart_command"] = restart.split()
        # Optional draft model: enables speculative decoding when the box is at the
        # single-stream bandwidth wall (the only lever that pushes tok/s past it).
        draft = ask(
            "  Draft .gguf for speculative decoding (small same-family quant) []: "
        ).strip()
        if draft:
            cfg.knobs["draft_model"] = draft
            dn = ask("    Draft tokens per step [16]: ").strip()
            if dn:
                try:
                    cfg.knobs["draft_n"] = int(dn)
                except ValueError:
                    pass

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
