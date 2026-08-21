"""Local embedding layer — deliberately no network in the hot path.

Model choice: **all-MiniLM-L6-v2** (sentence-transformers).
Why this one:
- 384-dim, ~80MB, 22M params — smallest model that still scores well on
  English STS/retrieval benchmarks; keeps P50 latency budget realistic on CPU.
- Trained on 1B+ English pairs incl. MS MARCO — our corpus IS MS MARCO
  derived, so domain fit is direct rather than hopeful.
- Loads once into process memory (`lru_cache`) and encodes locally; only the
  very first call hits the network to fetch weights.

Embeddings are L2-normalized so cosine distance == euclidean ranking; the
store uses cosine space and reports `score = 1 - distance` as raw similarity.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    model = SentenceTransformer(EMBEDDING_MODEL)
    # Guard against EMBEDDING_DIM drifting from the real model if it's swapped
    loaded = model.get_embedding_dimension()
    assert loaded == EMBEDDING_DIM, (
        f"{EMBEDDING_MODEL} produces {loaded}-dim vectors but EMBEDDING_DIM={EMBEDDING_DIM}"
    )
    return model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed a batch of texts; order-preserving, L2-normalized."""
    if not texts:
        return []
    vectors = _model().encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()
