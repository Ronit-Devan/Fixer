"""``et-remediation`` standalone CLI: setup, inspect config, flip the kill-switch.

This drives the *persisted* configuration and audit file. Live approve/reject of
disruptive actions happens against a running monitor's HTTP API (the manager
holds the in-memory approval state); this CLI covers the ops surface that lives
on disk: first-run setup, showing/changing the mode, and tailing the audit log.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from et_remediation.config import (
    RemediationConfig,
    RemediationMode,
    default_config_path,
)
from et_remediation.setup_ui import run_setup


def _cfg_path(args: argparse.Namespace) -> Path:
    return Path(args.config) if args.config else default_config_path()


def _cmd_setup(args: argparse.Namespace) -> int:
    path = _cfg_path(args)
    existing = RemediationConfig.load_or_default(path if path.is_file() else None)
    run_setup(path=path, existing=existing)
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    cfg = RemediationConfig.load_or_default(_cfg_path(args))
    print(json.dumps(cfg.to_dict(), indent=2))
    return 0


def _cmd_mode(args: argparse.Namespace) -> int:
    path = _cfg_path(args)
    cfg = RemediationConfig.load_or_default(path)
    try:
        cfg.mode = RemediationMode(args.mode)
    except ValueError:
        print(f"invalid mode {args.mode!r}; choose one of: "
              f"{', '.join(m.value for m in RemediationMode)}", file=sys.stderr)
        return 2
    cfg.configured = True
    cfg.save(path)
    print(f"mode -> {cfg.mode.value}  ({path})")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    cfg = RemediationConfig.load_or_default(_cfg_path(args))
    if not cfg.audit_path or not Path(cfg.audit_path).is_file():
        print("(no audit file configured or it does not exist yet)")
        return 0
    lines = Path(cfg.audit_path).read_text(encoding="utf-8").splitlines()
    for line in lines[-args.limit :]:
        print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="et-remediation",
        description="Configure the ET GPU remediation layer (kill-switch, caps, audit).",
    )
    parser.add_argument("--config", default="", help="Path to remediation config JSON.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="Interactive first-run setup wizard.")
    sub.add_parser("show", help="Print the current persisted configuration.")
    p_mode = sub.add_parser("mode", help="Set the operating mode (the kill-switch).")
    p_mode.add_argument("mode", choices=[m.value for m in RemediationMode])
    p_audit = sub.add_parser("audit", help="Tail the audit log (if a file sink is set).")
    p_audit.add_argument("--limit", type=int, default=20)

    args = parser.parse_args(argv)
    return {
        "setup": _cmd_setup,
        "show": _cmd_show,
        "mode": _cmd_mode,
        "audit": _cmd_audit,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
