"""Strategy registry — the single place query/index code goes to get a chunker."""

from app.chunking.base import Chunker
from app.chunking.fixed_size import FixedSizeChunker
from app.chunking.fixed_size_overlap import FixedSizeOverlapChunker
from app.chunking.metadata_aware import MetadataAwareChunker
from app.chunking.semantic import SemanticChunker

STRATEGIES: dict[str, Chunker] = {
    "fixed_size": FixedSizeChunker(),
    "fixed_size_overlap": FixedSizeOverlapChunker(),
    "semantic": SemanticChunker(),
    "metadata_aware": MetadataAwareChunker(inner=SemanticChunker()),
}


def get_chunker(name: str) -> Chunker:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise KeyError(
            f"unknown chunking strategy {name!r}; available: {sorted(STRATEGIES)}"
        ) from None
