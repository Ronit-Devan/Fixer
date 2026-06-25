"""/props parsing + the retry cap that keeps an absent /props from costing a
failing HTTP request every tick (the monitoring-overhead floor)."""

from __future__ import annotations

import et_monitor.llama as llama_mod
from et_monitor.llama import LlamaScraper, props_from_json


def test_props_from_json_tolerant_of_shapes():
    p = props_from_json(
        {"default_generation_settings": {"n_ctx": 8192}, "total_slots": 4, "model_path": "/m.gguf"}
    )
    assert p.reachable is True
    assert p.ctx_size == 8192
    assert p.total_slots == 4
    assert p.model_path == "/m.gguf"
    # Missing fields are simply None, never an error.
    assert p.cache_type_k is None


def test_read_props_stops_after_max_attempts(monkeypatch):
    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        raise OSError("no /props here")

    monkeypatch.setattr(llama_mod.urllib.request, "urlopen", _boom)
    sc = LlamaScraper("http://localhost:9", timeout_s=0.01)
    # Poll many ticks; the scraper must give up issuing HTTP after max_attempts.
    for _ in range(20):
        props = sc.read_props(max_attempts=3)
        assert props.reachable is False
    assert calls["n"] == 3  # never hammered once past the cap
