"""PARITY GATE for the torch -> ONNX embedding runtime swap.

The migration must prove numerical equivalence BEFORE embed.py changes:
the same fixed set of ~20 REAL corpus samples embedded through (a) the
current torch/sentence-transformers path and (b) the candidate direct-
onnxruntime path (transformers tokenizer + masked mean pooling + L2
normalize) must agree at cosine >= 0.999 on every sample.

If this fails, the swap does not happen — report and stop.

Skipped automatically where torch is absent (e.g. the slim runtime image
after migration); the gate's job is to run wherever both runtimes exist.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from app.config import get_settings

torch = pytest.importorskip("torch")  # noqa: F401 — gate on torch presence


def _real_samples(n_passages: int = 12, n_queries: int = 8) -> list[str]:
    """Fixed set of REAL texts: corpus passages + real dataset queries."""
    passages_path = Path(get_settings().processed_data_dir) / "passages.jsonl"
    samples: list[str] = []
    with open(passages_path) as f:
        for line in f:
            if len(samples) >= n_passages:
                break
            text = json.loads(line)["text"].strip()
            if text:
                samples.append(text)
    assert len(samples) == n_passages, "corpus file missing/short — parity set invalid"

    from app.benchmarking.queries import load_real_queries

    samples.extend(load_real_queries(n_queries))
    samples.append("what is the process of incorporation of a company")  # WARMUP_QUERY twin
    return samples


def _torch_reference(texts: list[str]) -> np.ndarray:
    """The CURRENT production path, verbatim semantics (embed.py pre-swap)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2", backend="torch")
    return model.encode(
        texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    )


def _onnx_candidate(texts: list[str]) -> np.ndarray:
    """The LIVE production path (app.retrieval.embed). This test doubles as a
    permanent regression guard: whenever torch is present locally, production
    vectors must keep matching the torch reference."""
    from app.retrieval.embed import embed_texts

    return np.asarray(embed_texts(texts), dtype=np.float32)


def test_onnx_matches_torch_on_real_corpus():
    texts = _real_samples()
    ref = _torch_reference(texts)
    cand = _onnx_candidate(texts)

    assert ref.shape == cand.shape == (len(texts), 384)

    cosines = (ref * cand).sum(axis=1)  # both L2-normalized -> dot == cosine
    print("\nparity cosines: min=%.6f mean=%.6f" % (cosines.min(), cosines.mean()))
    worst_idx = int(cosines.argmin())
    print("worst sample:", round(float(cosines[worst_idx]), 6), repr(texts[worst_idx][:60]))

    assert float(cosines.min()) > 0.999, (
        f"PARITY GATE FAILED: min cosine {float(cosines.min()):.6f} <= 0.999 "
        "— do not swap embed.py; investigate before proceeding"
    )


def test_candidate_is_l2_normalized():
    vecs = _onnx_candidate(_real_samples())
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
