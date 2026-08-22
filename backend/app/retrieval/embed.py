"""Local embedding layer — deliberately no network in the hot path.

Model choice: **all-MiniLM-L6-v2** (unchanged weights, unchanged rationale):
- 384-dim, ~90MB ONNX, 22M params — smallest model that still scores well on
  English STS/retrieval benchmarks; keeps P50 latency budget realistic on CPU.
- Trained on 1B+ English pairs incl. MS MARCO — our corpus IS MS MARCO
  derived, so domain fit is direct rather than hopeful.

Inference runtime: **direct ONNX Runtime** (swapped from torch/
sentence-transformers, Aug 2026). Same official weights from the model repo
(`onnx/model.onnx`); the swap exists because torch's runtime idles at
~480MB anonymous RSS — over small-container budgets — while ORT serves
numerically identical vectors at a fraction of that. Equivalence is gated
by tests/test_embedding_parity.py (cosine >= 0.999 on real corpus samples;
measured min = 1.000000). Pooling replicated exactly per the model's
config: attention-masked mean pooling, then L2 normalize.

Embeddings are L2-normalized so cosine distance == euclidean ranking; the
store uses cosine space and reports `score = 1 - distance` as raw similarity.
"""

from functools import lru_cache

import numpy as np
import onnxruntime as ort

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_FILE = "onnx/model.onnx"
EMBEDDING_DIM = 384
MAX_SEQ_LEN = 256  # mirrors the model's tokenizer_config


@lru_cache(maxsize=1)
def _session() -> tuple[ort.InferenceSession, object]:
    """Load tokenizer + ORT session once. First call may hit disk cache only —
    deployment images bake these files at build time (no network at boot)."""
    from transformers import AutoTokenizer

    from huggingface_hub import hf_hub_download

    onnx_path = hf_hub_download(EMBEDDING_MODEL, ONNX_FILE)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    return session, tokenizer


def _assert_dim(vectors: np.ndarray) -> None:
    loaded = vectors.shape[-1]
    assert loaded == EMBEDDING_DIM, (
        f"{EMBEDDING_MODEL} produces {loaded}-dim vectors but EMBEDDING_DIM={EMBEDDING_DIM}"
    )


def _encode(texts: list[str]) -> np.ndarray:
    """Tokenize -> ORT -> masked mean pool -> L2 normalize. Mirrors the
    sentence-transformers pipeline this replaced (see module docstring)."""
    session, tokenizer = _session()
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN,
        return_tensors="np",
    )
    feeds = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
        "token_type_ids": enc["token_type_ids"].astype(np.int64),
    }
    hidden = session.run(["last_hidden_state"], feeds)[0]  # (B, T, H)
    mask = enc["attention_mask"][:, :, None].astype(np.float32)
    pooled = (hidden * mask).sum(axis=1) / np.clip(
        mask.sum(axis=1), a_min=1e-9, a_max=None
    )
    normed = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
    _assert_dim(normed)
    return normed.astype(np.float32)


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed a batch of texts; order-preserving, L2-normalized."""
    if not texts:
        return []
    out = [_encode(batch) for batch in
           (texts[i:i + batch_size] for i in range(0, len(texts), batch_size))]
    return np.concatenate(out, axis=0).tolist()
