import pytest

from app.chunking.router import STRATEGIES, get_chunker


def test_all_four_strategies_registered():
    assert set(STRATEGIES) == {
        "fixed_size",
        "fixed_size_overlap",
        "semantic",
        "metadata_aware",
    }


def test_get_chunker_returns_working_chunkers():
    from app.ingestion.models import RawDocument

    doc = RawDocument(id="d1", text="Hello there world. Another sentence here.", metadata={})
    for name in STRATEGIES:
        chunker = get_chunker(name)
        chunks = chunker.chunk(doc)
        assert chunks, f"{name} produced no chunks on a valid document"
        assert all(c.metadata["strategy"] for c in chunks)


def test_unknown_strategy_raises_helpful_error():
    with pytest.raises(KeyError, match="no_such_strategy"):
        get_chunker("no_such_strategy")
