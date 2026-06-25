"""Bounded resource use over a long run: audit-file rotation + approvals pruning."""

from __future__ import annotations

from conftest import diag

from et_remediation import (
    LlamaCppActuator,
    RemediationConfig,
    RemediationManager,
    RemediationMode,
    default_registry,
)
from et_remediation.audit import AuditLog, AuditRecord, Phase
from et_remediation.engine import ApprovalRequest


def _rec(i: int) -> AuditRecord:
    return AuditRecord(
        ts=float(i), phase=Phase.APPLY, node_id="n", job_id=None, root_cause="rc",
        verdict="v", action_kind="k", action_class="non_disruptive", mode="auto",
        decision="applied", detail={"i": i, "pad": "x" * 200},
    )


def test_audit_jsonl_rotates_and_is_bounded(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(jsonl_path=path, max_bytes=2_000, backup_count=3)
    for i in range(500):  # far past the size cap
        log.record(_rec(i))
    # Current file + at most `backup_count` rotated files; nothing unbounded.
    rotated = sorted(tmp_path.glob("audit.jsonl.*"))
    assert path.exists()
    assert len(rotated) <= 3
    assert not (tmp_path / "audit.jsonl.4").exists()
    # The current file stays under the cap (it rotated before exceeding it badly).
    assert path.stat().st_size < 2_000 * 2
    # In-memory ring is independently bounded.
    assert len(log) <= 1000


def test_audit_no_path_is_noop(tmp_path):
    log = AuditLog()  # in-memory only
    for i in range(50):
        log.record(_rec(i))
    assert len(log) == 50  # ring keeps them; no file written


def _mgr(tmp_path):
    cfg = RemediationConfig(mode=RemediationMode.AUTO, knobs={"model": "m.gguf"})
    return RemediationManager(default_registry(), cfg, [LlamaCppActuator()], now_fn=lambda: 0.0)


def test_resolved_approvals_are_pruned_pending_kept(tmp_path):
    mgr = _mgr(tmp_path)
    mgr._approval_history_cap = 10
    # Stuff 200 resolved approvals + 1 pending.
    for i in range(200):
        mgr.approvals[f"a{i}"] = ApprovalRequest(
            id=f"a{i}", node_id="n", job_id=None, root_cause="rc", verdict="v",
            kind=next(iter(mgr.registry.all_specs())).kind, summary="s",
            command_preview="", requires_checkpoint=False, requires_drain=True,
            created_at=float(i), status="applied",
        )
    mgr.approvals["pending"] = ApprovalRequest(
        id="pending", node_id="n", job_id=None, root_cause="rc", verdict="v",
        kind=next(iter(mgr.registry.all_specs())).kind, summary="s", command_preview="",
        requires_checkpoint=False, requires_drain=True, created_at=999.0, status="pending",
    )
    mgr._prune_approvals()
    resolved = [a for a in mgr.approvals.values() if a.status != "pending"]
    assert len(resolved) == 10                      # capped
    assert "pending" in mgr.approvals               # pending never dropped
    # The most-recent resolved are the ones kept (highest created_at).
    assert min(int(a.id[1:]) for a in resolved) == 190


def test_observe_keeps_approvals_bounded_over_many_episodes(tmp_path):
    mgr = _mgr(tmp_path)
    mgr._approval_history_cap = 20
    # Each episode opens an approval, then it's "resolved" so the next opens anew.
    for i in range(300):
        out = mgr.observe(diag("gpu_offload_partial", metrics={"mem_used_ratio": 0.4}),
                          [], now=float(i))
        if out.approval is not None:
            out.approval.status = "applied"  # simulate operator resolving it
    assert len(mgr.approvals) <= 20 + 1  # bounded (history cap + at most one pending)
