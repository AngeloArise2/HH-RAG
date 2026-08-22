"""Tests for the remote (deployed-URL) latency benchmark plumbing."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench_remote  # noqa: E402


def _ok(embed=5.0, search=10.0, retrieval=15.0, gen=500.0, guard=900.0, total=1420.0):
    return {
        "refused": False,
        "latency_trace_ms": {
            "embed_query": embed, "vector_search": search,
            "generation": gen, "guardrail_check": guard,
        },
        "retrieval_ms": retrieval, "total_ms": total,
    }


def test_refused_queries_excluded_from_rows_but_counted(monkeypatch):
    responses = [_ok(), {"refused": True, "refusal_reason": "off_topic"}, _ok()]
    monkeypatch.setattr(bench_remote, "fetch", lambda client, url, q: responses.pop(0))
    result = bench_remote.run(client=None, base_url="http://x", queries=["a", "b", "c"], sleep_s=0)
    assert result["n_sent"] == 3
    assert result["n_refused"] == 1
    assert len(result["rows"]) == 2


def test_report_percentiles_over_collected_rows():
    rows = [
        {"embed_query": e, "vector_search": 10.0, "retrieval_ms": 10.0 + e, "total_ms": 1000.0}
        for e in [4.0, 5.0, 6.0, 8.0]
    ]
    table = bench_remote.report({"rows": rows})
    assert table["retrieval_ms"]["n"] == 4
    assert table["retrieval_ms"]["p50"] == pytest.approx(15.0)  # sorted 14,15,16,18
    assert table["embed_query"]["p100"] == pytest.approx(8.0)


def test_report_skips_missing_stages():
    rows = [{"embed_query": 5.0}]  # no generation key at all (e.g. refusal-free short path)
    table = bench_remote.report({"rows": rows})
    assert "generation" not in table
    assert table["embed_query"]["n"] == 1
