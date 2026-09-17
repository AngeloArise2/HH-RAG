# Architecture notes

See README.md for the diagram. Add deeper design notes here as decisions get made (e.g. why a given chunking strategy was set as default, embedding model tradeoffs, etc.).

## Dataset schema: ai4bharat/MSMARCO-XI (as observed on the real data)

The dataset does **not** ship a flat text corpus. Each row is one *query* with
nested passage lists:

```
query_id      int64
query_type    string   (e.g. description / entity / numeric)
query         string   (translated query in the shard's target language)
Eng_Query     string
Answer        string
Eng_Answer    string
source_lang / target_lang  string
passages:
    English_passages     list[string]   <- what we use
    Translated_passages  list[string]
    is_selected          list[int64]    (index-aligned with the passage lists;
                                         1 = this passage answers the query,
                                         MS MARCO style)
```

The repo itself is split into per-language parquet shards (`train/hintrain.parquet`,
`validation/hinval.parquet`, ...), ~4GB per train shard and ~460MB per validation
shard; total ~55GB over 10M train rows. We therefore read exactly one shard and
cap how many query rows we consume (`MAX_RAW_ROWS`, default 1000 — at ~10
passages per query row this lands ~10k processed passages, inside the
5k–20k working-set target).

### Choices made for this build

- **English_passages only.** We flatten each row's `English_passages` list into
  individual standalone passages (one row → N documents) and ignore
  `Translated_passages`. Rationale: time constraint plus a simpler pipeline —
  the default local sentence-transformers embedder planned for Phase 3 is an
  English model, so an English corpus avoids adding a multilingual embedding
  dependency to the hot path. The language shard choice (default `hin`) then
  only affects which queries surround the corpus, not corpus content.
- **Metadata preserved for future evaluation:** every passage document carries
  its source `query_id`, `is_selected` flag, `query_type`, `language`, `split`,
  and `passage_index` through preprocessing into `processed/passages.jsonl`.
  `is_selected` gives us free relevance labels — enough to build a small
  retrieval-quality eval later without any extra labeling work.
- **Stable ids:** `doc_id = sha1(cleaned_text)[:16]`, assigned at preprocess
  time, so identical passages dedupe across query rows and ids survive re-runs.

## Guardrail failure policy: fail-open, surfaced — a deliberate decision

When the grounding-judge LLM call fails (Groq 429s, timeouts, empty completions
— all observed live during phase 8 benchmarking), the pipeline does NOT block:
the generated answer is still returned with `AskResponse.grounding_verified=False`
and the real exception message preserved in `warnings[]`, instead of refusing.
This is availability-over-hard-blocking by choice: a transient hosted-API
outage in a *checker* should not take down answers from a pipeline whose other
stages succeeded, and silently dropping the answer would be indistinguishable
from an outage of the whole system. The caveat is structured, not string-matched:
`grounding_verified` and `refused` are independent fields covering different
failure modes — `refused=True` means a guard deliberately blocked an answer;
`grounding_verified=False` means an answer IS returned but was never judged.
They are not two readings of one state, and `grounding_verified` carries no
meaning when `refused=True` (nothing was generated, so nothing needed judging).
The input filter has no equivalent flag because its fail-open simply lets the
request proceed normally — the same tradeoff, applied where the cost is lower.

## Embedding inference runtime: torch -> ONNX Runtime (Aug 2026, parity-gated)

Post-deployment memory measurement forced a runtime swap: the container idled
at ~483MB *anonymous* RSS — almost entirely the torch runtime that
sentence-transformers imports — over small-container budgets (Render free
caps at 512MB; Railway trial instances price by RAM). The fix keeps the MODEL
and WEIGHTS identical and swaps only the inference engine:

- Weights (original): the official `onnx/model.onnx` shipped in the
  `sentence-transformers/all-MiniLM-L6-v2` HF repo (90MB), not a custom export.
- Runtime: `onnxruntime` CPU + `transformers` tokenizer; pooling replicated
  exactly per model config (attention-masked mean pool, L2 normalize).
- `sentence-transformers`/torch are gone from requirements AND the image —
  ST v6 imports torch eagerly even with `backend="onnx"`, so keeping the
  package would have kept the 480MB.

Gate (original swap): `backend/tests/test_embedding_parity.py` embedded ~21
real corpus passages/queries through BOTH paths and asserted per-sample
cosine >= 0.999. Measured at swap time: **min = 1.000000** (identical to 6
decimals). The test skips where torch is absent and doubled as a standing
regression guard wherever torch exists.

Measured effect (same cgroup methodology, idle after warmup): anon
**483MB -> 283MB (-41%)**, image **3.04GB -> 1.61GB**, embed_query p50
**~14ms -> ~6ms** (ORT is also faster on CPU for this model size).

## Multilingual attempt and revert (Sep 2026): all-MiniLM-L6-v2 -> paraphrase-multilingual-MiniLM-L12-v2 -> all-MiniLM-L6-v2

Attempted to let a user speak or type any Indic language (Sarvam STT
transcribes 22) and get answers from the English corpus by swapping the
English-only embedder for a **cross-lingual model**:

- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (~118MB uint8-quantized ONNX, AVX2 variant `onnx/model_quint8_avx2.onnx`).
  12-layer multilingual BERT. Supports 50+ languages including all 22 Indic
  languages Sarvam transcribes. A Hindi query about "incorporation" landed
  near the English passage about incorporation (top cosine 0.60 vs ~0.02 for
  an unrelated topic) — the cross-lingual retrieval worked.
- **Chunking (kept):** semantic chunker sentence-terminator regex extended
  with Devanagari danda (।) and double danda (॥). Model-agnostic, retained.
- **Threading (kept):** offline `build_index.py` sets `EMBED_THREADS=12`
  (overrideable) to parallelize the bulk re-embed; runtime stays pinned to 1.

**Why it was reverted — measured memory budget, Render free tier 512MB:**
the multilingual model structurally did not fit. Step-by-step live `VmRSS`
measurements (raw `/proc/self/status`, not `ru_maxrss`):

| component | RSS (~) |
|---|---|
| Python + FastAPI + chromadb + onnxruntime imports | 80MB |
| ONNX session with `enable_cpu_mem_arena=False`, BASIC graph opt | 144MB |
| multilingual tokenizer vocab (250K tokens, `tokenizers` Rust trie) | 270MB |
| Chroma loading a real collection | 112MB |
| **total** | **~600MB** |

Attempts that did NOT save enough: (a) raw `tokenizers.Tokenizer` instead of
`transformers.AutoTokenizer` — the slow Python tokenizer's overhead (~190MB)
was removed, but the Rust trie for 250K tokens alone held ~270MB; (b) ORT
memory-arena off (`enable_cpu_mem_arena=False`) + `ORT_ENABLE_BASIC` graph
opt — saved ~270MB of cached workspace on the QDQ forward path, a genuine
win kept for the English model too. Even both together left ~600MB. A ~100MB
tokenizer-vocab trim (drop CJK/Cyrillic/Arabic scripts, ~80K tokens) was
rejected: it lands *at* the cap and changes query tokenization vs the
full-vocab build index, a quality risk not worth a marginal fit.

**Reverted to all-MiniLM-L6-v2.** The English-only model with the retained
optimizations idles ~300MB (in-budget) and the deploy is green again.
Multilingual support is documented here as gated on a >1GB deployment (Render
starter+ / railway premium) or a future lower-vocab cross-lingual model.
`backend/tests/test_embedding_parity.py` now asserts the model id so a silent
model swap breaks CI instead of the deployment.

## Embedding runtime optimizations (kept from the multilingual attempt)

- **Raw `tokenizers` Rust tokenizer instead of `transformers.AutoTokenizer`.**
  `AutoTokenizer` builds a slow Python tokenizer alongside the fast Rust one,
  a large extra RSS slab; the raw `Tokenizer.from_file(tokenizer.json)`
  replaces it. Same tokens, leaner memory. Applies to any WordPiece model
  whose repo ships `tokenizer.json`.
- **`opts.enable_cpu_mem_arena = False`** plus
  `ORT_ENABLE_BASIC` graph optimization. ORT's default arena caches freed
  workspace after the first forward; disabling it returns that block to the
  OS. Small per-request allocator cost, worth it under a 512MB ceiling.
