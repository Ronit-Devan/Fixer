"""Snapshot prompt/cache facts must come only from an ACTIVELY PROCESSING slot:
/slots keeps reporting the last finished task (task_prev), and stale values on
idle ticks poison the analyzer window (a warm request right after a cold one
would inherit the cold request's zero-cache stats -> false prefill verdict)."""

from __future__ import annotations

from et_monitor.gpu import MockGpuSampler
from et_monitor.llama import SlotsInfo
from et_monitor.state import Monitor, MonitorConfig


class _StubScraper:
    """Minimal llama scraper: no /metrics, scripted /slots."""

    def __init__(self, slots: SlotsInfo | None) -> None:
        self.slots = slots

    def read(self):
        return None

    def read_slots(self):
        return self.slots


def _snap_for(slots: SlotsInfo | None):
    mon = Monitor(MockGpuSampler(), _StubScraper(slots), MonitorConfig(interval_s=1.0))
    return mon.tick()


def test_processing_slot_facts_are_captured():
    s = _snap_for(SlotsInfo(reachable=True, n_slots=1, processing=1,
                            prompt_tokens=30000, cache_tokens=29900))
    assert s.prompt_tokens == 30000.0
    assert s.cache_tokens == 29900.0


def test_stale_finished_slot_facts_are_dropped():
    # Same numbers but nothing processing (task_prev leftovers) -> not carried.
    s = _snap_for(SlotsInfo(reachable=True, n_slots=1, processing=0,
                            prompt_tokens=30000, cache_tokens=0))
    assert s.prompt_tokens is None
    assert s.cache_tokens is None


def test_no_slots_endpoint_is_fine():
    s = _snap_for(None)
    assert s.prompt_tokens is None and s.cache_tokens is None
