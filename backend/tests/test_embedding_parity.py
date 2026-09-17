"""Embedding behavior gate for the ONNX runtime.

History: this file was originally the torch->ONNX parity gate for
all-MiniLM-L6-v2 (same weights, different inference engine, cosine >= 0.999).
That gate served its purpose at the Aug 2026 swap. A later attempt to upgrade
to the multilingual model (paraphrase-multilingual-MiniLM-L12-v2) had to be
R E V E R T E D : its 250K-vocab tokenizer alone held ~270MB RSS, pushing the
full stack to ~600MB — over Render free tier's 512MB cap. See
docs/architecture.md for the measurements. English-only all-MiniLM idles at
~300MB with the same raw-tokenizer + no-arena optimizations that made the
attempt close (raw tokenizer + no-arena ORT session are KEPT for this model).

This file now guards the properties that matter for the English-only path:
  - model choice (regression gate against an accidental multilingual swap
    that would silently break the deployment's memory budget)
  - shape: 384-dim, L2-normalized output
  - determinism: same input -> same vector
  - semantic signal: a related English passage ranks above an unrelated one

Skipped where torch is absent — the remaining checks need no torch at all.
"""

import numpy as np
import pytest

from app.retrieval.embed import EMBEDDING_DIM, EMBEDDING_MODEL, embed_texts


def test_model_is_the_english_only_one():
    """Multilingual models blow the 512MB Render budget at runtime (measured).
    An unexpected model id here means the memory budget was re-validated and
    this gate updated deliberately — not silently."""
    assert EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_candidate_is_l2_normalized():
    samples = [
        "What is the process of incorporation of a company",
        "The sun is a star at the center of our solar system.",
        "Quantum entanglement links particles regardless of distance.",
    ]
    vecs = np.asarray(embed_texts(samples), dtype=np.float32)
    assert vecs.shape == (len(samples), EMBEDDING_DIM)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_deterministic_output():
    text = "What is the process of incorporation of a company"
    a = embed_texts([text])[0]
    b = embed_texts([text])[0]
    assert max(abs(x - y) for x, y in zip(a, b)) < 1e-4


def test_related_english_passage_outranks_unrelated():
    """The corpus is English-only, so retrieval value reduces to: a query must
    rank its matching passage above unrelated passages in embedding space."""
    query = "What is the process of incorporation of a company"
    match = (
        "The process of incorporation involves filing articles of incorporation "
        "with the state and paying the applicable filing fee."
    )
    unrelated = (
        "Sourdough bread needs a lively starter and a long bulk fermentation."
    )

    q = embed_texts([query])[0]
    m = embed_texts([match])[0]
    u = embed_texts([unrelated])[0]
    cos_match = float(np.dot(q, m))
    cos_unrelated = float(np.dot(q, u))
    assert cos_match > cos_unrelated, (
        f"semantic signal lost: match={cos_match:.4f} unrelated={cos_unrelated:.4f}"
    )