"""Local embedding layer — deliberately no network in the hot path.

Model choice: **paraphrase-multilingual-MiniLM-L12-v2** (quantized uint8):
- 384-dim, ~118MB ONNX (uint8 quantized, AVX2), 12-layer BERT — supports
  50+ languages incl. all 22 Indian languages that Sarvam STT transcribes.
- Cross-lingual similarity: a Hindi query about "incorporation" maps near an
  English passage about incorporation, enabling multilingual queries against
  our English-only corpus without changing the retrieval pipeline.
- Replaces all-MiniLM-L6-v2 (English-only, 6-layer) which blocked non-English
  queries from producing meaningful vector similarities.

Inference runtime: **direct ONNX Runtime** (swapped from torch/
sentence-transformers, Aug 2026). Same official weights from the model repo
(`onnx/model_quint8_avx2.onnx`); the quantized variant is chosen for broad
x86-64 CPU compatibility (AVX2 is universal since ~2013). Pooling replicated
exactly per the model's config: attention-masked mean pooling, then L2
normalize. max_seq_length set to 128 (the model's sentence-transformers
default) rather than 256 — passages and queries are short enough that this
is sufficient and saves forward-pass time.

Embeddings are L2-normalized so cosine distance == euclidean ranking; the
store uses cosine space and reports `score = 1 - distance` as raw similarity.
"""

from functools import lru_cache

import numpy as np
import onnxruntime as ort

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ONNX_FILE = "onnx/model_quint8_avx2.onnx"  # uint8 quantized, AVX2 — broad x86-64 compat
EMBEDDING_DIM = 384  # same dimension as the old model; no downstream shape changes
MAX_SEQ_LEN = 128   # multilingual model's sentence-transformers default (was 256 for English-only)


def _thread_count() -> int:
    """ORT intra-op thread count. Runtime hot path wants 1 (deterministic, and
    ORT auto-sizing overshoots throttled containers), but the offline index
    build benefits from parallelism. EMBED_THREADS overrides the default.
    """
    import os

    return max(1, int(os.environ.get("EMBED_THREADS", "1")))


@lru_cache(maxsize=1)
def _session() -> tuple[ort.InferenceSession, object]:
    """Load tokenizer + ORT session once. First call may hit disk cache only —
    deployment images bake these files at build time (no network at boot)."""
    from transformers import AutoTokenizer

    from huggingface_hub import hf_hub_download

    onnx_path = hf_hub_download(EMBEDDING_MODEL, ONNX_FILE)
    # Default to pinning thread pools to 1: ORT auto-sizes from the HOST core
    # count, which wildly overshoots a throttled container (Render free = 0.1
    # shared CPU) — pool spin-up plus cross-thread contention inflated
    # embed_query several-fold there. Single-threaded is also deterministic.
    # The offline build path sets EMBED_THREADS (e.g. 12) for parallelism.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = _thread_count()
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )
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
