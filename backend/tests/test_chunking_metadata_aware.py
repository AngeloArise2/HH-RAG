from app.chunking.metadata_aware import MetadataAwareChunker
from app.chunking.semantic import SemanticChunker
from app.ingestion.models import RawDocument


def _doc() -> RawDocument:
    text = "Alpha sentence one. Beta sentence two. Gamma sentence three. Delta four."
    return RawDocument(
        id="src-doc-9",
        text=text,
        metadata={
            "query_id": 12345,
            "query_type": "description",
            "is_selected": True,
            "language": "hin",
            "split": "validation",
            "source_dataset": "ai4bharat/MSMARCO-XI",
            "passage_index": 2,
            # field NOT in DEFAULT_DOC_FIELDS — must not leak into chunks:
            "secret_noise": "x",
        },
    )


def test_enriches_metadata_with_document_fields():
    wrapped = MetadataAwareChunker(inner=SemanticChunker())
    chunks = wrapped.chunk(_doc())
    assert chunks
    for c in chunks:
        assert c.metadata["strategy"] == "metadata_aware"
        assert c.metadata["inner_strategy"] == "semantic"
        assert c.metadata["doc_query_id"] == 12345
        assert c.metadata["doc_query_type"] == "description"
        assert c.metadata["doc_is_selected"] is True
        assert c.metadata["doc_language"] == "hin"
        assert c.metadata["doc_split"] == "validation"


def test_unlisted_doc_fields_do_not_leak():
    wrapped = MetadataAwareChunker(inner=SemanticChunker())
    for c in wrapped.chunk(_doc()):
        assert "doc_secret_noise" not in c.metadata


def test_chunk_texts_identical_to_unwrapped_inner():
    doc = _doc()
    inner_chunks = SemanticChunker().chunk(doc)
    wrapped_chunks = MetadataAwareChunker(inner=SemanticChunker()).chunk(doc)
    assert [c.text for c in wrapped_chunks] == [c.text for c in inner_chunks]
    assert [c.start_offset for c in wrapped_chunks] == [c.start_offset for c in inner_chunks]
    assert [c.end_offset for c in wrapped_chunks] == [c.end_offset for c in inner_chunks]


def test_chunk_ids_reminted_under_wrapper_strategy():
    wrapped = MetadataAwareChunker(inner=SemanticChunker())
    chunks = wrapped.chunk(_doc())
    assert chunks
    for c in chunks:
        assert c.chunk_id.startswith(f"{c.doc_id}-metadata_aware-")
        assert "-semantic-" not in c.chunk_id
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)  # unique within document


def test_deterministic_output():
    run_a = MetadataAwareChunker(inner=SemanticChunker()).chunk(_doc())
    run_b = MetadataAwareChunker(inner=SemanticChunker()).chunk(_doc())
    assert [c.model_dump() for c in run_a] == [c.model_dump() for c in run_b]


def test_custom_field_selection():
    wrapped = MetadataAwareChunker(inner=SemanticChunker(), fields=["language"])
    for c in wrapped.chunk(_doc()):
        assert "doc_query_id" not in c.metadata
        assert c.metadata["doc_language"] == "hin"
