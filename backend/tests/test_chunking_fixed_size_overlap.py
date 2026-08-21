from app.chunking.fixed_size_overlap import FixedSizeOverlapChunker
from app.ingestion.models import RawDocument


def _make_doc(n_words: int, id: str = "doc1") -> RawDocument:
    return RawDocument(id=id, text=" ".join(f"w{i:04d}" for i in range(n_words)), metadata={})


# window=10, ratio=0.2 -> overlap=2 words, step=8
def _chunker() -> FixedSizeOverlapChunker:
    return FixedSizeOverlapChunker(max_words=10, overlap_ratio=0.2)


def test_adjacent_chunks_share_exactly_the_configured_overlap():
    doc = _make_doc(40)
    chunks = _chunker().chunk(doc)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:]):
        tail = a.text.split()[-2:]
        head = b.text.split()[:2]
        assert tail == head, f"overlap broken: ...{tail!r} vs {head!r}"


def test_no_chunk_exceeds_max_length_and_none_empty():
    doc = _make_doc(40)
    chunks = _chunker().chunk(doc)
    assert chunks
    for c in chunks:
        assert c.text.strip()
        assert len(c.text.split()) <= 10


def test_full_coverage_first_to_last_word():
    doc = _make_doc(40)
    chunks = _chunker().chunk(doc)
    assert len(chunks) > 1
    assert chunks[0].text.split()[0] == "w0000"
    assert chunks[-1].text.split()[-1] == "w0039"
    # every source word survives in at least one chunk
    chunk_words = {w for c in chunks for w in c.text.split()}
    assert set(doc.text.split()) <= chunk_words


def test_offsets_slice_back_to_source_text():
    doc = _make_doc(40)
    for c in _chunker().chunk(doc):
        assert c.text == doc.text[c.start_offset : c.end_offset]


def test_deterministic_chunk_count():
    doc = _make_doc(40)
    run_a = _chunker().chunk(doc)
    run_b = _chunker().chunk(doc)
    assert [c.model_dump() for c in run_a] == [c.model_dump() for c in run_b]


def test_overlap_ratio_bounds_enforced():
    import pytest

    with pytest.raises(ValueError):
        FixedSizeOverlapChunker(max_words=10, overlap_ratio=1.0)
