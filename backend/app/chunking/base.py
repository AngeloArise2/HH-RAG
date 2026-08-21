"""Shared contracts for every chunking strategy.

Conventions all strategies must honor:
- `start_offset`/`end_offset` are character spans into the ORIGINAL document
  text, so `chunk.text == document.text[start_offset:end_offset]` always holds
  (grounding/citations later depend on this).
- `chunk_id` is deterministic given (document id, strategy, position).
- metadata carries at least {"strategy", "position"}; strategies may add more.
- Empty or whitespace-only documents produce [] — never empty chunks.
"""

import re
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.ingestion.models import RawDocument
from app.ingestion.preprocess import stable_doc_id


class Chunk(BaseModel):
    text: str
    doc_id: str
    chunk_id: str
    start_offset: int
    end_offset: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunker(Protocol):
    def chunk(self, document: RawDocument) -> list[Chunk]: ...


_WORD_RE = re.compile(r"\S+")


def word_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) char offsets of every whitespace-delimited word."""
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def resolve_doc_id(document: RawDocument) -> str:
    """Raw downloads have no id yet; fall back to the same content hash
    preprocessing uses so ids stay stable across pipeline stages."""
    return document.id if document.id is not None else stable_doc_id(document.text)


def make_chunks(
    document: RawDocument, *, strategy: str, pieces: list[tuple[int, int]]
) -> list[Chunk]:
    """Turn (start_char, end_char) spans into Chunk objects."""
    doc_id = resolve_doc_id(document)
    chunks: list[Chunk] = []
    for i, (start, end) in enumerate(pieces):
        text = document.text[start:end]
        if not text.strip():
            continue  # defensive; strategies should never emit empty spans
        chunks.append(
            Chunk(
                text=text,
                doc_id=doc_id,
                chunk_id=f"{doc_id}-{strategy}-{i}",
                start_offset=start,
                end_offset=end,
                metadata={"strategy": strategy, "position": i},
            )
        )
    return chunks
