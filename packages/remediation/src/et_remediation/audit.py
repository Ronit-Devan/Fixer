"""Append-only audit trail of every remediation decision.

The design requires a *full* audit log: trigger, decision, apply, verify result,
and rollback. Every state change in the manager emits one ``AuditRecord``. The
log keeps a bounded in-memory ring (so the dashboard can show recent activity)
and optionally mirrors every record to a JSONL file for durable, greppable ops
history. Writing is best-effort: a failed disk write never breaks remediation.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


class Phase(str, Enum):
    """Which step in the lifecycle this record captures."""

    TRIGGER = "trigger"  # a diagnosis arrived and was mapped to a root cause
    DECISION = "decision"  # policy/breaker decided allow/block/advise
    APPLY = "apply"  # a non-disruptive action was applied (or dry-run built)
    VERIFY = "verify"  # recovery confirmed within the window
    ROLLBACK = "rollback"  # recovery failed -> auto-reverted
    APPROVAL_REQUESTED = "approval_requested"  # disruptive: human gate opened
    APPROVAL_APPLIED = "approval_applied"  # disruptive: human approved + ran
    BLOCKED = "blocked"  # breaker open / rate cap / kill-switch / caps
    ADVISE = "advise"  # advise-only plan emitted, no actuation


@dataclass(frozen=True)
class AuditRecord:
    ts: float
    phase: Phase
    node_id: str
    job_id: str | None
    root_cause: str
    verdict: str
    action_kind: str | None
    action_class: str | None
    mode: str
    decision: str  # short machine-readable outcome, e.g. "applied", "rate_capped"
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["phase"] = self.phase.value
        return d


class AuditLog:
    """Bounded in-memory ring + optional JSONL mirror."""

    def __init__(self, *, maxlen: int = 1000, jsonl_path: str | Path | None = None) -> None:
        self._records: deque[AuditRecord] = deque(maxlen=maxlen)
        self._path = Path(jsonl_path) if jsonl_path else None
        if self._path is not None:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:  # pragma: no cover - defensive
                log.warning("audit: cannot create %s: %s", self._path.parent, e)
                self._path = None

    def record(self, rec: AuditRecord) -> AuditRecord:
        self._records.append(rec)
        if self._path is not None:
            try:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec.to_dict()) + "\n")
            except OSError as e:  # pragma: no cover - defensive
                log.warning("audit: append to %s failed: %s", self._path, e)
        return rec

    def recent(self, limit: int | None = None) -> list[AuditRecord]:
        items = list(self._records)
        return items[-limit:] if limit else items

    def as_dicts(self, limit: int | None = None) -> list[dict]:
        return [r.to_dict() for r in self.recent(limit)]

    def __iter__(self) -> Iterable[AuditRecord]:  # type: ignore[override]
        return iter(list(self._records))

    def __len__(self) -> int:
        return len(self._records)
