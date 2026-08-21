from app.chunking.fixed_size import FixedSizeChunker
from app.ingestion.models import RawDocument


def _make_doc(n_words: int, id: str = "doc1") -> RawDocument:
    return RawDocument(id=id, text=" ".join(f"w{i:04d}" for i in range(n_words)), metadata={})


def test_no_empty_chunks_and_max_length_respected():
    doc = _make_doc(120)
    chunks = FixedSizeChunker(max_words=50).chunk(doc)
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert c.text.strip()
        assert len(c.text.split()) <= 50


def test_exact_word_partition():
    doc = _make_doc(120)
    chunks = FixedSizeChunker(max_words=50).chunk(doc)
    assert [len(c.text.split()) for c in chunks] == [50, 50, 20]
    # no gaps/overlaps between consecutive chunks
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_offset >= a.end_offset


def test_offsets_slice_back_to_source_text():
    doc = _make_doc(120)
    for c in FixedSizeChunker(max_words=50).chunk(doc):
        assert c.text == doc.text[c.start_offset : c.end_offset]


def test_deterministic_chunk_count_and_ids():
    doc = _make_doc(120, id="same-id")
    run_a = FixedSizeChunker(max_words=50).chunk(doc)
    run_b = FixedSizeChunker(max_words=50).chunk(doc)
    assert len(run_a) == len(run_b)
    assert [c.model_dump() for c in run_a] == [c.model_dump() for c in run_b]
    assert {c.chunk_id for c in run_a} == {"same-id-fixed_size-0", "same-id-fixed_size-1", "same-id-fixed_size-2"}


def test_document_shorter_than_max_yields_single_chunk():
    chunks = FixedSizeChunker(max_words=100).chunk(_make_doc(10))
    assert len(chunks) == 1


def test_empty_document_yields_no_chunks():
    assert FixedSizeChunker().chunk(RawDocument(id="d", text="   ", metadata={})) == []


def test_invalid_config_rejected():
    import pytest

    with pytest.raises(ValueError):
        FixedSizeChunker(max_words=0)
