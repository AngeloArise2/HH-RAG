import pytest

from app.chunking.semantic import SemanticChunker
from app.ingestion.models import RawDocument


def _doc(text: str, id: str = "doc1") -> RawDocument:
    return RawDocument(id=id, text=text, metadata={})


SHORT_SENT = "The quick brown fox jumps over the lazy dog near the riverbank at dawn."
# 14 words per sentence; max_words=30 -> two sentences per group


def test_respects_soft_max_and_never_cuts_mid_sentence():
    text = " ".join(f"Sent{i} {SHORT_SENT}" for i in range(6))
    chunks = SemanticChunker(max_words=30).chunk(_doc(text))
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.text.strip()
        if not c.metadata.get("oversized_sentence"):
            assert len(c.text.split()) <= 30
        stripped = c.text.rstrip()
        if i < len(chunks) - 1:
            assert stripped[-1] in ".!?", f"chunk {i} cut mid-sentence: ...{stripped[-40:]!r}"


def test_oversized_sentence_emitted_alone_not_split():
    long_sentence = " ".join(f"w{i}" for i in range(50)) + "."  # 51 words, no internal punct
    text = f"{SHORT_SENT} {long_sentence} {SHORT_SENT}"
    chunks = SemanticChunker(max_words=30).chunk(_doc(text))
    oversized = [c for c in chunks if c.metadata.get("oversized_sentence")]
    assert len(oversized) == 1
    assert oversized[0].text.strip() == long_sentence.strip()


def test_offsets_slice_back_to_source_text():
    text = f"{SHORT_SENT} {SHORT_SENT.replace('fox', 'cat')} {SHORT_SENT}"
    for c in SemanticChunker(max_words=30).chunk(_doc(text)):
        assert c.text == text[c.start_offset : c.end_offset]


def test_deterministic_chunk_count():
    text = " ".join(SHORT_SENT for _ in range(10))
    run_a = SemanticChunker(max_words=30).chunk(_doc(text))
    run_b = SemanticChunker(max_words=30).chunk(_doc(text))
    assert [c.model_dump() for c in run_a] == [c.model_dump() for c in run_b]


def test_no_terminal_punctuation_still_one_chunk():
    chunks = SemanticChunker(max_words=100).chunk(_doc("words without any punctuation " * 5))
    assert len(chunks) == 1


def test_empty_document_yields_no_chunks():
    assert SemanticChunker().chunk(_doc("")) == []


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        SemanticChunker(max_words=0)
