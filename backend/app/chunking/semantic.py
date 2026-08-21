"""Semantic-ish strategy: split on sentence boundaries, group sentences up to
a SOFT max length rather than cutting mid-sentence.

A single sentence longer than max_words is emitted alone (flagged
`oversized_sentence` in its metadata) instead of being hard-split — losing a
few oversized chunks beats shredding a sentence the retriever needs intact.
"""

import re

from app.chunking.base import Chunk, make_chunks
from app.ingestion.models import RawDocument

STRATEGY_NAME = "semantic"

# Lazily match up to a punctuation cluster (+ trailing whitespace) or EOF.
# Keeps offsets exact: each match span slices cleanly out of the source text.
_SENTENCE_RE = re.compile(r".+?(?:[.!?]+(?:\s+|$)|$)", re.DOTALL)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _SENTENCE_RE.finditer(text) if m.group(0).strip()]


class SemanticChunker:
    def __init__(self, max_words: int = 120) -> None:
        if max_words < 1:
            raise ValueError("max_words must be >= 1")
        self.max_words = max_words

    def chunk(self, document: RawDocument) -> list[Chunk]:
        text = document.text
        pieces: list[tuple[int, int]] = []
        oversized_flags: list[bool] = []

        current_start: int | None = None
        prev_end = 0
        current_words = 0

        def flush() -> None:
            assert current_start is not None
            pieces.append((current_start, prev_end))
            oversized_flags.append(current_words > self.max_words)

        for start, end in sentence_spans(text):
            n_words = len(text[start:end].split())

            if current_start is not None and current_words + n_words > self.max_words:
                flush()
                current_start = None

            if current_start is None:
                current_start = start
                current_words = n_words
            else:
                current_words += n_words
            prev_end = end

            # A lone over-long sentence: emit alone, never grow the group.
            if current_words > self.max_words:
                flush()
                current_start = None

        if current_start is not None:
            flush()

        chunks = make_chunks(document, strategy=STRATEGY_NAME, pieces=pieces)
        for chunk, oversized in zip(chunks, oversized_flags):
            if oversized:
                chunk.metadata["oversized_sentence"] = True
        return chunks
