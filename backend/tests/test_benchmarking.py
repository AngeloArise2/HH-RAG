"""Phase 8 benchmarking machinery tests: percentile math, query loading,
startup warmup.

Offline-first: dataset/index-dependent tests skip loudly when local assets
are missing rather than failing CI on environment differences.
"""

from pathlib import Path

import pytest

from app.benchmarking.latency import percentile, summarize
from app.config import get_settings


# --- percentile() ------------------------------------------------------------


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_percentile_single_element():
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 99) == 42.0


def test_percentile_odd_count_exact_median():
    vals = [3.0, 1.0, 2.0]  # unsorted input must be handled
    assert percentile(vals, 50) == 2.0


def test_percentile_interpolates_between_points():
    # p70 over [0, 100] -> rank = 0.7 * 1 = 0.7 -> 70.0
    assert percentile([0.0, 100.0], 70) == 70.0
    # p50 over two points lands exactly halfway between them
    assert percentile([10.0, 20.0], 50) == 15.0


def test_percentile_bounds():
    vals = [5.0, 1.0, 9.0, 3.0]
    assert percentile(vals, 0) == 1.0
    assert percentile(vals, 100) == 9.0


# --- summarize() -------------------------------------------------------------


def test_summarize_shape_and_values():
    s = summarize([10.0, 20.0, 30.0, 40.0])
    assert s["n"] == 4
    assert s["min"] == 10.0
    assert s["p100"] == 40.0
    assert s["p50"] == 25.0
    assert s["mean"] == 25.0
    assert {"n", "min", "p50", "p70", "p100", "mean"} <= set(s)


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize([])


# --- load_real_queries ---------------------------------------------------------

_HF_SNAPSHOTS = (
    Path.home() / ".cache" / "huggingface" / "hub"
    / "datasets--ai4bharat--MSMARCO-XI/snapshots"
)
DATASET_DOWNLOADED = _HF_SNAPSHOTS.exists()


@pytest.mark.skipif(not DATASET_DOWNLOADED, reason="dataset not downloaded")
def test_load_real_queries_returns_distinct_real_questions():
    from app.benchmarking.queries import load_real_queries

    queries = load_real_queries(7)
    assert len(queries) == 7
    assert len(set(queries)) == 7
    assert all(len(q.strip()) > 10 for q in queries)


def test_load_real_queries_zero_is_empty_without_touching_disk():
    from app.benchmarking.queries import load_real_queries

    assert load_real_queries(0) == []


# --- warm_retrieval --------------------------------------------------------------

def _index_ready() -> bool:
    try:
        from app.retrieval.vector_store import collection_counts

        settings = get_settings()
        return collection_counts(settings).get(settings.default_chunk_strategy, 0) > 0
    except Exception:
        return False


INDEX_READY = _index_ready()


@pytest.mark.skipif(not INDEX_READY, reason="vector index not built")
def test_warm_retrieval_returns_positive_ms_and_is_repeatable():
    from app.retrieval.retriever import warm_retrieval

    first = warm_retrieval()
    second = warm_retrieval()
    assert first > 0.0
    assert second > 0.0
    # second call must not pay the model-load cost again
    assert second < max(first, 5000.0)
