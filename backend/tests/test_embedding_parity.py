"""Embedding behavior gate for the multilingual ONNX runtime.

History: this file was originally the torch->ONNX parity gate for
all-MiniLM-L6-v2 (same weights, different inference engine, cosine >= 0.999).
That gate served its purpose at the Aug 2026 swap and was later superseded:
we then replaced the English-only model with the multilingual model
(paraphrase-multilingual-MiniLM-L12-v2) to support Indic-language queries,
which is a WEIGHTS change, not a runtime change — torch-parity no longer
applies because the reference model id itself changed.

This file now guards the properties that matter for the multilingual path:
  - shape: 384-dim, L2-normalized output
  - determinism: same input -> same vector
  - cross-lingual signal: a Hindi query lands closer to the matching English
    passage than to an unrelated English passage (the whole point of the swap)
Skipped where torch is absent — the remaining checks need no torch at all.
"""

import numpy as np
import pytest

from app.retrieval.embed import EMBEDDING_DIM, embed_texts


def test_candidate_is_l2_normalized():
    samples = [
        "What is the process of incorporation of a company",
        "सूरज एक तारा है",  # "the sun is a star"
        "Quantum entanglement links particles regardless of distance.",
    ]
    vecs = np.asarray(embed_texts(samples), dtype=np.float32)
    assert vecs.shape == (len(samples), EMBEDDING_DIM)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_deterministic_output():
    text = "कंपनी के पंजीकरण की प्रक्रिया क्या है"
    a = embed_texts([text])[0]
    b = embed_texts([text])[0]
    assert max(abs(x - y) for x, y in zip(a, b)) < 1e-4


def test_cross_lingual_query_matches_its_english_passage():
    """A Hindi query must rank its matching English passage above unrelated ones.
    This is the behaviour the English-only model could NOT provide, so it is
    the regression gate for the whole multilingual swap."""
    hindi_query = "कंपनी के पंजीकरण की प्रक्रिया क्या है"  # "what is the process of incorporating a company"
    english_match = (
        "The process of incorporation involves filing articles of incorporation "
        "with the state and paying the applicable filing fee."
    )
    english_unrelated = (
        "Sourdough bread needs a lively starter and a long bulk fermentation."
    )

    q = embed_texts([hindi_query])[0]
    match = embed_texts([english_match])[0]
    unrelated = embed_texts([english_unrelated])[0]
    cos_match = float(np.dot(q, match))
    cos_unrelated = float(np.dot(q, unrelated))
    assert cos_match > cos_unrelated, (
        f"cross-lingual signal lost: match={cos_match:.4f} unrelated={cos_unrelated:.4f}"
    )
