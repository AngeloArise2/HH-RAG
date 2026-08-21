"""Decorator strategy: wraps another chunker and enriches every chunk's
metadata with document-level fields (source row info from the dataset), so
retrieval can filter/boost on them later without touching chunk texts.

MSMARCO-XI passages carry no title/section fields, so the useful document-
level surface here is the source row: query_id, query_type, is_selected,
language, split, source_dataset. Fields are prefixed `doc_` to keep them
collision-free against chunk-level keys like `strategy`.
"""

from collections.abc import Iterable

from app.chunking.base import Chunk
from app.ingestion.models import RawDocument

STRATEGY_NAME = "metadata_aware"

DEFAULT_DOC_FIELDS = (
    "query_id",
    "query_type",
    "is_selected",
    "language",
    "split",
    "source_dataset",
    "passage_index",
)


class MetadataAwareChunker:
    def __init__(self, inner, fields: Iterable[str] | None = None) -> None:
        self.inner = inner
        self.fields = frozenset(fields) if fields is not None else frozenset(DEFAULT_DOC_FIELDS)

    def chunk(self, document: RawDocument) -> list[Chunk]:
        chunks = self.inner.chunk(document)
        doc_fields = {k: v for k, v in document.metadata.items() if k in self.fields}
        for c in chunks:
            c.metadata["inner_strategy"] = c.metadata.get("strategy")
            c.metadata["strategy"] = STRATEGY_NAME
            # re-mint chunk_id under OUR registered name (inner ids say e.g.
            # "-semantic-") so ids stay collision-free if strategies ever share
            # one store; position index is already in metadata from the inner
            c.chunk_id = f"{c.doc_id}-{STRATEGY_NAME}-{c.metadata['position']}"
            c.metadata.update({f"doc_{k}": v for k, v in doc_fields.items()})
        return chunks
