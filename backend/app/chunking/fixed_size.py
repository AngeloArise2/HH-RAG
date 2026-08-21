"""Baseline strategy: fixed number of words per chunk, no overlap.

Deliberately naive — it exists so the smarter strategies have something to
beat in the Phase 8 retrieval comparison.
"""

from app.chunking.base import Chunk, make_chunks, word_spans
from app.ingestion.models import RawDocument

STRATEGY_NAME = "fixed_size"


class FixedSizeChunker:
    def __init__(self, max_words: int = 100) -> None:
        if max_words < 1:
            raise ValueError("max_words must be >= 1")
        self.max_words = max_words

    def chunk(self, document: RawDocument) -> list[Chunk]:
        spans = word_spans(document.text)
        pieces = [
            (
                spans[i][0],
                spans[min(i + self.max_words, len(spans)) - 1][1],
            )
            for i in range(0, len(spans), self.max_words)
        ]
        return make_chunks(document, strategy=STRATEGY_NAME, pieces=pieces)
