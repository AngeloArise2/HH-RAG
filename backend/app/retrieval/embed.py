"""Local embedding layer — deliberately no network in the hot path.

Model choice: **all-MiniLM-L6-v2** (English-only, fp32):
- 384-dim, ~86MB ONNX (fp32), 6-layer BERT. English-only by design —
  deliberately chosen over paraphrase-multilingual-MiniLM-L12-v2 (50+
  languages, 250K-vocab tokenizer) because the multilingual variant's
  tokenizer alone held ~270MB RSS, pushing the full stack to ~600MB over
  Render free tier's 512MB cap. Measured, documented (see
  docs/architecture.md), reverted. Multilingual needs a >1GB instance.

Inference runtime: **direct ONNX Runtime** (swapped from torch/
sentence-transformers, Aug 2026). Same official weights from the model repo
(`onnx/model.onnx`, fp32). Pooling replicated exactly per the model's config:
attention-masked mean pooling, then L2 normalize. max_seq_length 256 (the
model's sentence-transformers default).

Tokenizer: uses the raw `tokenizers` Rust library directly instead of
`transformers.AutoTokenizer` — the latter loads a slow Python tokenizer
alongside the fast Rust one. Raw loads leaner and avoids the ~190MB
transformers overhead (critical inside Render's 512MB cap).

Embeddings are L2-normalized so cosine distance == euclidean ranking; the
store uses cosine space and reports `score = 1 - distance` as raw similarity.
"""

from functools import lru_cache

import numpy as np
import onnxruntime as ort

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_FILE = "onnx/model.onnx"  # official fp32 export
EMBEDDING_DIM = 384
MAX_SEQ_LEN = 256   # English-only model's sentence-transformers default


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
    from tokenizers import Tokenizer

    from huggingface_hub import hf_hub_download

    onnx_path = hf_hub_download(EMBEDDING_MODEL, ONNX_FILE)
    tok_path = hf_hub_download(EMBEDDING_MODEL, "tokenizer.json")
    # Default to pinning thread pools to 1: ORT auto-sizes from the HOST core
    # count, which wildly overshoots a throttled container (Render free = 0.1
    # shared CPU) — pool spin-up plus cross-thread contention inflated
    # embed_query several-fold there. Single-threaded is also deterministic.
    # The offline build path sets EMBED_THREADS (e.g. 12) for parallelism.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = _thread_count()
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    # Disable the CPU memory arena: ORT keeps freed workspace cached in the
    # default arena, which held a large block of RSS after the first forward
    # (~270MB with the heavier multilingual model). Disabling it releases
    # that memory, essential for staying under Render free tier's 512MB cap.
    opts.enable_cpu_mem_arena = False
    session = ort.InferenceSession(
        str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    # Use the raw Rust tokenizer — transformers.AutoTokenizer loads a slow
    # Python tokenizer alongside the fast one, adding ~190MB for 250K vocab.
    tok = Tokenizer.from_file(str(tok_path))
    tok.enable_truncation(max_length=MAX_SEQ_LEN)
    tok.enable_padding(
        length=None,       # dynamic: pad to longest in batch
        pad_id=0,
        pad_token="[PAD]",
    )
    return session, tok


def _assert_dim(vectors: np.ndarray) -> None:
    loaded = vectors.shape[-1]
    assert loaded == EMBEDDING_DIM, (
        f"{EMBEDDING_MODEL} produces {loaded}-dim vectors but EMBEDDING_DIM={EMBEDDING_DIM}"
    )


def _encode(texts: list[str]) -> np.ndarray:
    """Tokenize -> ORT -> masked mean pool -> L2 normalize. Mirrors the
    sentence-transformers pipeline this replaced (see module docstring)."""
    session, tok = _session()
    encodings = tok.encode_batch(texts)
    ids = np.array([e.ids for e in encodings], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    types = np.array([e.type_ids for e in encodings], dtype=np.int64)
    feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
    hidden = session.run(["last_hidden_state"], feeds)[0]  # (B, T, H)
    mask_f = mask[:, :, None].astype(np.float32)
    pooled = (hidden * mask_f).sum(axis=1) / np.clip(
        mask_f.sum(axis=1), a_min=1e-9, a_max=None
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
