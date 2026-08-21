"""Fixed-size chunks with a configurable overlap window.

Same word-window idea as fixed_size, but consecutive windows share
`overlap_ratio` of their words so an idea straddling a boundary appears
whole in at least one chunk.
"""

from app.chunking.base import Chunk, make_chunks, word_spans
from app.ingestion.models import RawDocument

STRATEGY_NAME = "fixed_size_overlap"


class FixedSizeOverlapChunker:
    def __init__(self, max_words: int = 100, overlap_ratio: float = 0.2) -> None:
        if max_words < 1:
            raise ValueError("max_words must be >= 1")
        if not 0 <= overlap_ratio < 1:
            raise ValueError("overlap_ratio must be in [0, 1)")
        self.max_words = max_words
        self.overlap_words = round(max_words * overlap_ratio)
        # step >= 1 guarantees forward progress and no duplicate final window
        self.step = max(1, max_words - self.overlap_words)

    def chunk(self, document: RawDocument) -> list[Chunk]:
        spans = word_spans(document.text)
        total = len(spans)
        if total == 0:
            return []

        pieces: list[tuple[int, int]] = []
        i = 0
        while True:
            j = min(i + self.max_words, total)
            pieces.append((spans[i][0], spans[j - 1][1]))
            if j >= total:
                break
            i += self.step
        return make_chunks(document, strategy=STRATEGY_NAME, pieces=pieces)
