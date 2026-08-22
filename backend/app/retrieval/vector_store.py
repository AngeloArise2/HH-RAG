"""Vector store layer: one Chroma collection per chunking strategy.

Persisted under VECTOR_STORE_PATH from config. Each strategy gets its own
collection (same corpus, different segmentation), which is what lets Phase 8
compare retrieval quality/latency per strategy instead of guessing.

Scores: embeddings are L2-normalized and collections use cosine space, so
`score = 1 - distance` lands in [-1, 1] and reads as cosine similarity.
"""

from typing import Any
import logging

import chromadb
from chromadb.errors import NotFoundError
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.chunking.base import Chunk
from app.retrieval.embed import embed_texts

logger = logging.getLogger(__name__)

# Chroma caps how many rows a single add/upsert can carry; stay well under it.
_UPSERT_BATCH = 2000


class RetrievedChunk(BaseModel):
    """One search hit, structured so callers never touch raw chroma dicts."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


def _client(settings: Settings | None = None) -> chromadb.api.ClientAPI:
    settings = settings or get_settings()
    # settings.vector_store_path is typed str; tolerate Path objects too.
    return chromadb.PersistentClient(path=str(settings.vector_store_path))


# Collection handles cached per (store_path, name). Fetching one costs sqlite
# reads + segment validation on every call; measured live on throttled CPU
# (Render free), that per-request machinery dominated vector_search (~700ms
# vs ~23ms locally). The client's heavy internals are shared by chromadb,
# but the handle itself was being rebuilt per request — same class of bug as
# the embedder singleton. Keyed by path so tests with tmp stores stay isolated.
_collections: dict[tuple[str, str], Any] = {}


def _collection(name: str, settings: Settings | None = None) -> Any:
    settings = settings or get_settings()
    key = (str(settings.vector_store_path), name)
    handle = _collections.get(key)
    if handle is None:
        handle = _client(settings).get_collection(name=name)
        _collections[key] = handle
    return handle


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    # Chroma accepts only str/int/float/bool metadata values; drop None/other.
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}


def build_index(
    chunks: list[Chunk], strategy_name: str, settings: Settings | None = None
) -> int:
    """(Re)build the collection for one strategy; returns chunk count indexed."""
    if not chunks:
        raise ValueError(f"no chunks provided for strategy {strategy_name!r}")
    collection = _client(settings).get_or_create_collection(
        name=strategy_name,
        metadata={"hnsw:space": "cosine"},
    )
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    for i in range(0, len(chunks), _UPSERT_BATCH):
        sl = slice(i, i + _UPSERT_BATCH)
        metadatas = [
            _sanitize_metadata({"doc_id": c.doc_id, **c.metadata}) for c in chunks[sl]
        ]
        collection.upsert(
            ids=[c.chunk_id for c in chunks[sl]],
            embeddings=vectors[sl],
            documents=texts[sl],
            metadatas=metadatas,
        )
    # Report what the store actually holds, not what we asked it to hold —
    # duplicate chunk_ids would silently collapse under upsert.
    return collection.count()


def collection_counts(settings: Settings | None = None) -> dict[str, int]:
    """Vector count per existing collection — for scripts/monitoring."""
    client = _client(settings)
    return {col.name: client.get_collection(col.name).count() for col in client.list_collections()}


def query(
    text: str,
    strategy_name: str,
    top_k: int = 5,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Convenience one-shot: embed + search in a single call.

    Prefer Retriever (retrieval/retriever.py) on the request path — it times
    embed_query and vector_search as separate stages.
    """
    return search_vectors(embed_texts([text])[0], strategy_name, top_k, settings=settings)


def search_vectors(
    query_vector: list[float],
    strategy_name: str,
    top_k: int = 5,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Top-k similar chunks for an ALREADY-EMBEDDED query vector, best first."""
    settings = settings or get_settings()
    try:
        collection = _collection(strategy_name, settings)
    except NotFoundError:
        # not-yet-built collection: no results beats a crash. Logged loudly
        # because a silently-empty store is indistinguishable from a working
        # one downstream — this exact silence once masked a cwd-dependent
        # path misconfiguration (every query returned zero context).
        logger.warning(
            "collection %r not found in %s — returning 0 hits "
            "(index not built or store path misconfigured)",
            strategy_name,
            settings.vector_store_path,
        )
        return []

    total = collection.count()
    if total == 0:
        return []
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, total),
        include=["documents", "metadatas", "distances"],
    )
    hits: list[RetrievedChunk] = []
    ids = response["ids"][0]
    docs = response["documents"][0]
    dists = response["distances"][0]
    metas = response["metadatas"][0]
    for chunk_id, text_hit, dist, meta in zip(ids, docs, dists, metas):
        meta = dict(meta or {})
        hits.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=str(meta.pop("doc_id", "")),
                text=text_hit,
                score=float(1.0 - dist),
                metadata=meta,
            )
        )
    return hits
