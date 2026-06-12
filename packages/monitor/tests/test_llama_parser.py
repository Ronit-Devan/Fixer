"""Prometheus exposition parsing for llama-server /metrics."""

from __future__ import annotations

from et_monitor.llama import metrics_from_raw, parse_prometheus

SAMPLE = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 1024
# HELP llamacpp:tokens_predicted_total Number of generation tokens processed.
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 512
llamacpp:kv_cache_usage_ratio 0.42
llamacpp:requests_processing 2
llamacpp:requests_deferred 1
llamacpp:predicted_tokens_seconds 73.5
some_metric{label="x",other="y"} 9.0
malformed_line_without_value
broken_value foo
"""


def test_parse_basic():
    raw = parse_prometheus(SAMPLE)
    assert raw["llamacpp:prompt_tokens_total"] == 1024
    assert raw["llamacpp:tokens_predicted_total"] == 512
    assert raw["llamacpp:kv_cache_usage_ratio"] == 0.42
    assert raw["llamacpp:requests_processing"] == 2
    assert raw["llamacpp:requests_deferred"] == 1


def test_parse_strips_labels():
    raw = parse_prometheus(SAMPLE)
    assert raw["some_metric"] == 9.0


def test_parse_skips_malformed():
    raw = parse_prometheus(SAMPLE)
    assert "malformed_line_without_value" not in raw
    assert "broken_value" not in raw


def test_parse_empty():
    assert parse_prometheus("") == {}
    assert parse_prometheus("# only comments\n") == {}


def test_metrics_from_raw_accessors():
    raw = parse_prometheus(SAMPLE)
    m = metrics_from_raw(raw, timestamp_s=1.0)
    assert m.reachable
    assert m.requests_processing == 2
    assert m.is_active
    assert m.kv_cache_usage_ratio == 0.42


def test_is_active_false_when_idle():
    m = metrics_from_raw({"llamacpp:requests_processing": 0}, timestamp_s=1.0)
    assert not m.is_active
