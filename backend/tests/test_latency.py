"""Tests for latency.py + Retriever per-stage instrumentation.

The retriever tests monkeypatch embed/search with fakes of known cost —
we're testing that stages get recorded SEPARATELY and honestly, not the
model's speed (that's what scripts/time_retrieval.py measures for real).
"""

import pytest

from app.benchmarking.latency import (
    EMBED_QUERY,
    RETRIEVAL_STAGES,
    VECTOR_SEARCH,
    LatencyTrace,
    current_trace,
    reset_current_trace,
    set_current_trace,
    stage_timer,
)


def test_stage_timer_records_into_explicit_trace():
    trace = LatencyTrace()
    with stage_timer(EMBED_QUERY, trace):
        sum(i * i for i in range(2000))
    assert EMBED_QUERY in trace.stage_ms
    assert trace.stage_ms[EMBED_QUERY] > 0


def test_repeated_stages_accumulate():
    trace = LatencyTrace()
    for _ in range(3):
        with stage_timer("generation", trace):
            pass
    assert len(trace.stage_ms) == 1
    assert trace.stage_ms["generation"] >= 0


def test_records_even_when_body_raises():
    trace = LatencyTrace()
    with pytest.raises(RuntimeError):
        with stage_timer(VECTOR_SEARCH, trace):
            raise RuntimeError("llm exploded")
    assert VECTOR_SEARCH in trace.stage_ms


def test_no_trace_anywhere_is_a_no_op_not_crash():
    with stage_timer(EMBED_QUERY):  # no explicit trace, no current trace
        pass
    assert current_trace() is None


def test_request_context_trace():
    trace = LatencyTrace()
    token = set_current_trace(trace)
    try:
        with stage_timer("stt"):
            pass
        with stage_timer("vector_search"):
            pass
    finally:
        reset_current_trace(token)
    assert set(trace.stage_ms) == {"stt", "vector_search"}


def test_retrieval_ms_sums_only_retrieval_stages():
    trace = LatencyTrace(stage_ms={"embed_query": 10.0, "vector_search": 5.0, "stt": 900.0, "generation": 700.0})
    assert trace.retrieval_ms() == 15.0
    assert trace.total_ms() == 1615.0
    assert set(RETRIEVAL_STAGES) == {"embed_query", "vector_search", "chunk_assembly"}


# --- Retriever instrumentation (faked embed/search with distinct costs) ---

@pytest.fixture
def faked_costs(monkeypatch):
    """Patch embed+search with controllable fake latencies (ms)."""
    costs = {"embed": 7.0, "search": 3.0}

    def fake_embed(texts, **kwargs):
        import time

        time.sleep(costs["embed"] / 1000)
        return [[0.1] * 384 for _ in texts]

    def fake_search(vector, strategy_name, top_k=5, settings=None):
        import time

        time.sleep(costs["search"] / 1000)
        from app.retrieval.vector_store import RetrievedChunk

        return [RetrievedChunk(chunk_id="c0", doc_id="d0", text="hit", score=0.9, metadata={})]

    from app.retrieval import embed as embed_module
    from app.retrieval import vector_store

    monkeypatch.setattr(embed_module, "embed_texts", fake_embed)
    monkeypatch.setattr(vector_store, "search_vectors", fake_search)
    return costs


def test_retriever_times_stages_separately(faked_costs):
    from app.config import Settings
    from app.retrieval.retriever import Retriever

    settings = Settings(default_chunk_strategy="semantic")
    result = Retriever(settings=settings).retrieve("test query")

    e = result.trace.stage_ms["embed_query"]
    s = result.trace.stage_ms["vector_search"]
    # fakes sleep ~7ms vs ~3ms; if stages were merged or mislabeled this fails
    assert e > faked_costs["search"]  # embed clearly larger than search would be
    assert s < e
    assert abs(result.trace.retrieval_ms() - (e + s)) < 2.0  # only these two stages ran
    assert "chunk_assembly" not in result.trace.stage_ms


def test_retriever_uses_configured_strategy(faked_costs):
    from app.config import Settings
    from app.retrieval.retriever import Retriever

    settings = Settings(default_chunk_strategy="fixed_size_overlap")
    result = Retriever(settings=settings).retrieve("q")
    assert result.strategy == "fixed_size_overlap"
    assert result.chunks[0].chunk_id == "c0"
