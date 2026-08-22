"""Round-trip tests: chunk -> build_index -> query must return the right topic.

Uses the real local embedding model (all-MiniLM-L6-v2) — that's the point:
a fake embedder would make the round-trip test meaningless. First run
downloads ~80MB of weights; afterwards everything is local.
"""

import pytest

from app.chunking.base import Chunk
from app.config import Settings
from app.retrieval import vector_store
from app.retrieval.embed import EMBEDDING_DIM, embed_texts

BREAD_TEXTS = [
    "Sourdough bread needs a lively starter and a long bulk fermentation.",
    "Bake the loaf in a dutch oven at 240C to get a crackly crust.",
    "Bread flour with higher protein gives sourdough better structure.",
]
PHYSICS_TEXTS = [
    "Quantum entanglement links particles regardless of distance between them.",
    "The double slit experiment shows wave particle duality of electrons.",
    "Superconductors conduct electricity with zero electrical resistance below a critical temperature.",
]


def _chunk(doc_id: str, strategy: str, pos: int, text: str) -> Chunk:
    return Chunk(
        text=text,
        doc_id=doc_id,
        chunk_id=f"{doc_id}-{strategy}-{pos}",
        start_offset=0,
        end_offset=len(text),
        metadata={"strategy": strategy, "position": pos, "query_id": 1},
    )


@pytest.fixture(scope="module")
def index_settings(tmp_path_factory) -> Settings:
    """Build two tiny collections once; all queries in this module hit them."""
    settings = Settings(vector_store_path=str(tmp_path_factory.mktemp("index")))
    bread = [_chunk(f"bread{i}", "test_strategy", i, t) for i, t in enumerate(BREAD_TEXTS)]
    physics = [_chunk(f"phys{i}", "other_strategy", i, t) for i, t in enumerate(PHYSICS_TEXTS)]
    assert vector_store.build_index(bread, "test_strategy", settings=settings) == 3
    assert vector_store.build_index(physics, "other_strategy", settings=settings) == 3
    return settings


def test_embed_shapes_and_determinism():
    vecs = embed_texts(["hello world", "second text"])
    assert len(vecs) == 2
    assert all(len(v) == EMBEDDING_DIM for v in vecs)
    again = embed_texts(["hello world"])
    assert max(abs(a - b) for a, b in zip(vecs[0], again[0])) < 1e-4


def test_roundtrip_query_returns_matching_topic(index_settings):
    hits = vector_store.query(
        "how do I bake sourdough bread at home",
        "test_strategy",
        top_k=2,
        settings=index_settings,
    )
    assert len(hits) == 2
    joined = " ".join(h.text.lower() for h in hits)
    assert "sourdough" in joined or "bread" in joined
    # best hit should be about baking, not quantum mechanics
    top = hits[0].text.lower()
    assert any(kw in top for kw in ("sourdough", "bread", "bake"))


def test_collections_are_isolated_per_strategy(index_settings):
    hits = vector_store.query(
        "how do I bake sourdough bread at home",
        "other_strategy",  # physics-only collection
        top_k=3,
        settings=index_settings,
    )
    assert len(hits) == 3
    joined = " ".join(h.text.lower() for h in hits).split()
    # no bread vocabulary should surface from the physics collection's texts
    assert not {"sourdough", "dutch"} & set(joined)


def test_scores_ordered_best_first(index_settings):
    hits = vector_store.query(
        "particles and waves in modern physics",
        "other_strategy",
        top_k=3,
        settings=index_settings,
    )
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_unknown_collection_returns_empty_not_crash(index_settings):
    assert vector_store.query("anything", "no_such_collection", settings=index_settings) == []


def test_build_index_reports_true_store_count(index_settings):
    # Within one upsert batch chroma rejects duplicate ids loudly
    # (DuplicateIDError). Silent collapse happens ACROSS calls/batches when a
    # reused chunk_id overwrites an old row -- so the reported count must come
    # from the store, not from len(chunks).
    def make(cid: str, text: str) -> Chunk:
        return Chunk(
            text=text,
            doc_id="doc-x",
            chunk_id=cid,
            start_offset=0,
            end_offset=len(text),
            metadata={"strategy": "dupe_strategy", "position": 0},
        )

    first = [make("dupe-0", "original chunk about sourdough bread.")]
    assert vector_store.build_index(first, "dupe_strategy", settings=index_settings) == 1

    # same id reused (text changed) + one genuinely new id
    second = [
        make("dupe-0", "rewritten chunk about bread."),
        make("dupe-1", "brand new chunk about physics."),
    ]
    reported = vector_store.build_index(second, "dupe_strategy", settings=index_settings)
    assert reported == 2  # store holds 2 rows, not 3
    assert vector_store.collection_counts(index_settings)["dupe_strategy"] == 2


def test_build_index_rejects_empty_chunk_list(index_settings):
    import pytest

    with pytest.raises(ValueError):
        vector_store.build_index([], "empty_ok", settings=index_settings)


def test_collection_handle_reused_across_queries(index_settings, monkeypatch):
    """The hot path must not reopen the client/collection per request —
    per-request fetches cost sqlite reads + segment validation every call
    (measured ~700ms vector_search on throttled CPU vs ~23ms locally)."""
    calls = {"n": 0}
    real_client = vector_store._client

    def counting_client(settings=None):
        calls["n"] += 1
        return real_client(settings)

    monkeypatch.setattr(vector_store, "_client", counting_client)
    # cold start: earlier tests in this module already warmed the cache
    vector_store._collections.pop((str(index_settings.vector_store_path), "test_strategy"), None)
    vec = embed_texts(["how do I bake sourdough bread at home"])[0]
    vector_store.search_vectors(vec, "test_strategy", top_k=2, settings=index_settings)
    vector_store.search_vectors(vec, "test_strategy", top_k=2, settings=index_settings)
    assert calls["n"] == 1  # second query reused the cached handle


def test_cached_handle_stays_correct_after_upsert(index_settings):
    """Cache must serve CURRENT store contents: upsert new rows, then query."""
    vec = embed_texts(["brand new sentence about superconductors and resistance."])[0]
    before = vector_store.search_vectors(vec, "other_strategy", top_k=3, settings=index_settings)
    chunk = _chunk("phys-new", "other_strategy", 9,
                   "A new physics passage about superconductor resistance.")
    assert vector_store.build_index(
        [*[_chunk(f"phys{i}", "other_strategy", i, t) for i, t in enumerate(PHYSICS_TEXTS)], chunk],
        "other_strategy",
        settings=index_settings,
    ) == 4
    after = vector_store.search_vectors(vec, "other_strategy", top_k=3, settings=index_settings)
    assert any("superconductor" in h.text.lower() for h in after[:2])
    assert len(before) == 3 and len(after) == 3
