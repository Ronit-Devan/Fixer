"""/slots parsing — the prefix-cache reuse signal (n_prompt_tokens_cache) that
replaced the removed /metrics KV counters — and the failure cap that stops
polling a build without /slots."""

from __future__ import annotations

import et_monitor.llama as llama_mod
from et_monitor.llama import LlamaScraper, SlotsInfo, slots_from_json


def test_slots_from_json_summarises_busiest_slot_and_hit_rate():
    obj = [
        {"id": 0, "n_ctx": 32768, "is_processing": True, "speculative": False,
         "n_prompt_tokens": 30000, "n_prompt_tokens_cache": 29900,
         "n_prompt_tokens_processed": 100},
        {"id": 1, "n_ctx": 32768, "is_processing": False,
         "n_prompt_tokens": 50, "n_prompt_tokens_cache": 0},
    ]
    s = slots_from_json(obj)
    assert s.reachable and s.n_slots == 2 and s.processing == 1
    assert s.prompt_tokens == 30000 and s.cache_tokens == 29900
    assert s.n_ctx == 32768
    assert abs(s.prefix_hit_rate - 29900 / 30000) < 1e-6  # warm ~0.997


def test_slots_cold_prefix_low_hit_rate():
    obj = [{"id": 0, "n_ctx": 32768, "is_processing": True,
            "n_prompt_tokens": 30000, "n_prompt_tokens_cache": 0,
            "n_prompt_tokens_processed": 5000}]
    s = slots_from_json(obj)
    assert s.prefix_hit_rate == 0.0  # cold: nothing reused


def test_slots_from_json_tolerates_missing_fields_and_bad_shape():
    # Task-not-assigned slots omit cache/prompt fields; must not crash.
    assert slots_from_json([{"id": 0, "n_ctx": 4096, "is_processing": False}]).prompt_tokens is None
    assert slots_from_json({}).reachable is False  # not a list
    assert slots_from_json([]).reachable is True and slots_from_json([]).n_slots == 0
    assert SlotsInfo().prefix_hit_rate is None


def test_read_slots_stops_after_repeated_failures(monkeypatch):
    scraper = LlamaScraper("http://localhost:9")  # nothing listening

    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise OSError("refused")

    monkeypatch.setattr(llama_mod.urllib.request, "urlopen", boom)
    for _ in range(5):
        assert scraper.read_slots(max_failures=3) is None
    # Capped at 3 real attempts, then disabled -> no further HTTP calls.
    assert calls["n"] == 3
    assert scraper._slots_disabled is True
