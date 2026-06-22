"""Operator-facing configuration for the remediation layer.

One persisted file (JSON) captures every knob the setup UI writes: the operating
``mode`` (the global kill-switch), which action classes are enabled, the verify
window, and the circuit-breaker / blast-radius caps. The monitor and the CLI
both load this; the dashboard settings panel reads and writes it.

Until setup is completed (``configured == False``) the safe default is
ADVISE — the layer never actuates blind on a fresh box.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class RemediationMode(str, Enum):
    """The global operating mode — this is the kill-switch.

    OFF      remediation layer does nothing (pure observability).
    ADVISE   produce remediation *plans* only; never actuate (advise-only).
    DRY_RUN  build and log the exact real commands, but never execute them.
    AUTO     unattended auto-apply for the NON-DISRUPTIVE class, through the
             full guarded path (apply -> verify -> confirm/rollback). DISRUPTIVE
             actions are still only ever surfaced as approval requests.
    """

    OFF = "off"
    ADVISE = "advise"
    DRY_RUN = "dry_run"
    AUTO = "auto"

    @property
    def actuates(self) -> bool:
        """Does this mode ever run a real command?"""
        return self is RemediationMode.AUTO

    @property
    def considers_actions(self) -> bool:
        """Does this mode evaluate/plan actions at all?"""
        return self in (
            RemediationMode.ADVISE,
            RemediationMode.DRY_RUN,
            RemediationMode.AUTO,
        )


@dataclass
class CapsConfig:
    """Circuit-breaker thresholds and blast-radius limits."""

    # Rate cap: at most this many auto-applies per node within window_s.
    max_actions_per_window: int = 3
    window_s: float = 600.0
    # Flap: the same action kind re-triggering this many times within
    # flap_window_s on a node trips the breaker (a fix that won't stick).
    flap_threshold: int = 3
    flap_window_s: float = 900.0
    # Consecutive non-recoveries (failed verify -> rollback) before tripping.
    failure_threshold: int = 3
    # Once OPEN, wait this long before allowing a single half-open trial.
    breaker_cooldown_s: float = 1800.0
    # Blast radius: max concurrent in-flight (VERIFYING) actions per node.
    max_concurrent_per_node: int = 1


@dataclass
class RemediationConfig:
    """The full, persisted configuration."""

    mode: RemediationMode = RemediationMode.ADVISE
    # Per-class master switches (within AUTO, an operator can still disable a
    # whole class). DISRUPTIVE here only governs whether approval *requests* are
    # generated; it never permits auto-execution.
    enable_non_disruptive: bool = True
    enable_disruptive_requests: bool = True
    # Seconds to watch for recovery after an auto-applied non-disruptive fix.
    verify_window_s: float = 30.0
    # Minimum number of telemetry samples taken AFTER the fix before recovery may
    # be judged. Guards against confirming/rolling back on a single reading, and
    # (with post-apply filtering) against judging against pre-fix samples.
    min_verify_samples: int = 2
    # Require the SAME root cause for this many consecutive observations before
    # acting — debounces a one-tick noisy verdict so we never actuate on a blip.
    # 1 = act on first sight (library default, backward-compatible). Production
    # wiring raises this (paired with predictive detection's lead time).
    trigger_debounce: int = 1
    caps: CapsConfig = field(default_factory=CapsConfig)
    # The live workload to protect (PIDs and/or a scheduler label).
    protected_pids: list[int] = field(default_factory=list)
    protected_label: str = ""
    # Tuning knobs passed to specs (power headroom %, target nice, etc.).
    knobs: dict = field(default_factory=dict)
    # Append-only JSONL audit sink; None keeps audit in-memory only.
    audit_path: str | None = None
    # Has the operator completed setup? Gates the "advise until configured" rule.
    configured: bool = False

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mode"] = self.mode.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RemediationConfig":
        d = dict(d)
        mode = d.pop("mode", RemediationMode.ADVISE.value)
        caps = d.pop("caps", None)
        cfg = cls(mode=RemediationMode(mode), **d)
        if caps:
            cfg.caps = CapsConfig(**caps)
        return cfg

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RemediationConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def load_or_default(cls, path: str | Path | None) -> "RemediationConfig":
        """Load config if the file exists; otherwise the safe ADVISE default."""
        if path is None:
            return cls()
        p = Path(path)
        if not p.is_file():
            return cls()
        try:
            return cls.load(p)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            # A corrupt config must fail safe, not crash the monitor.
            return cls()


def default_config_path() -> Path:
    """Canonical location for the persisted config (~/.et/remediation.json)."""
    return Path.home() / ".et" / "remediation.json"
