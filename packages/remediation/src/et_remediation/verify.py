"""Recovery predicates for the verify-and-rollback loop.

After a non-disruptive fix is applied, the manager watches a bounded window of
telemetry and asks the spec's ``recovered(pre, post)`` whether the fix worked.
These are the reusable building blocks those predicates are made of. Each is
conservative: if the relevant signal is *missing* in either window, recovery is
NOT proven (returns False), so we never confirm a fix on absent data and instead
let it roll back.
"""

from __future__ import annotations

from et_remediation.telemetry import WindowSummary


def clock_recovered(
    pre: WindowSummary, post: WindowSummary, *, min_ratio: float = 0.80, min_gain: float = 0.05
) -> bool:
    """SM clock climbed back up (throttle relieved).

    Recovery must be *attributable to the fix*: the post window has to clear an
    absolute floor AND show a real gain over the throttled pre window. Requiring
    the gain (not just the floor) prevents a false CONFIRM when the post window
    merely *looks* healthy without the fix having moved anything — the pure
    absolute-floor test used to confirm a fix that did nothing.
    """
    if pre.mean_clock_ratio is None or post.mean_clock_ratio is None:
        return False
    return (
        post.mean_clock_ratio >= min_ratio
        and post.mean_clock_ratio - pre.mean_clock_ratio >= min_gain
    )


def temp_dropped(pre: WindowSummary, post: WindowSummary, *, min_drop_c: float = 2.0) -> bool:
    if pre.max_temp_c is None or post.max_temp_c is None:
        return False
    return pre.max_temp_c - post.max_temp_c >= min_drop_c


def util_recovered(
    pre: WindowSummary, post: WindowSummary, *, min_gain_pct: float = 10.0
) -> bool:
    """GPU utilization climbed after the fix (e.g. CPU bottleneck relieved)."""
    if pre.mean_util_pct is None or post.mean_util_pct is None:
        return False
    return post.mean_util_pct - pre.mean_util_pct >= min_gain_pct


def memory_freed(
    pre: WindowSummary, post: WindowSummary, *, min_drop_ratio: float = 0.10
) -> bool:
    """VRAM was reclaimed (e.g. an orphan holding the card was killed).

    Compares the post window MEAN (not its single-sample min) against the pre
    mean: a transient one-sample dip must not be mistaken for a real, sustained
    reclaim, or we would confirm a kill that freed nothing.
    """
    if pre.mean_mem_used_ratio is None or post.mean_mem_used_ratio is None:
        return False
    return pre.mean_mem_used_ratio - post.mean_mem_used_ratio >= min_drop_ratio
