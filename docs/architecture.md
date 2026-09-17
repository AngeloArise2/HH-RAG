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

## Multilingual embedding swap (Sep 2026): all-MiniLM-L6-v2 -> paraphrase-multilingual-MiniLM-L12-v2

The original English-only embedder (all-MiniLM-L6-v2) made the whole pipeline
effectively English-only: Sarvam's STT could transcribe 22 Indian languages,
but the English-only embeddings produced meaningless vector similarities for
non-English queries. To let a user speak or type any Indic language and get
answers from the English corpus, the embedder was swapped for a **cross-lingual
model**:

- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (~118MB uint8-quantized ONNX, AVX2 variant `onnx/model_quint8_avx2.onnx`).
  12-layer multilingual BERT. Supports 50+ languages including all 22 Indic
  languages Sarvam transcribes. Cross-lingual similarity: a Hindi query about
  "incorporation" lands near the English passage about incorporation.
- **Tokenizer max length lowered to 128** (the model's sentence-transformers
  default) from the old 256 — passages and queries are short enough that this
  is sufficient and trims forward-pass time.
- **Quantization choice:** uint8 (AVX2) chosen for broad x86-64 CPU
  compatibility (AVX2 is universal since ~2013). Faster AVX-512/VNNI variants
  exist but aren't safe on unknown deployment CPUs.
- **Chunking:** semantic chunker sentence-terminator regex extended with
  Devanagari danda (।) and double danda (॥) so it splits Indic-script
  sentences correctly.

Measured cross-lingual retrieval (real corpus, metadata_aware collection):
a Hindi query "कंपनी के पंजीकरण की प्रक्रिया क्या है" (what is the
incorporation process) retrieves relevant English registration passages
(top cosine 0.60) vs an unrelated-topic cosine of ~0.02. English queries
retrieve at similar quality to before (top 0.617). **Known limitation:**
cross-lingual quality is uneven — Hindi maps well to English; some less-well-
represented Indic languages (e.g. Bengali observed at cosine ~0.56 against an
unrelated passage) retrieve more noise. This is a model-quality constraint,
not a pipeline defect.

Index principle unchanged: this is a WEIGHTS change, not an architecture one.
The ONNX loading, mean-pooling, and L2-normalization path is identical to
before. Because vectors live in a new model's vector space, the index was
**fully rebuilt** (`scripts/build_index.py`, ~18 min at 12 embed threads) and
the deployment snapshot (`index_snapshot.tgz`) regenerated.

Threading note: the runtime hot path keeps ORT pinned to 1 thread
(`intra_op_num_threads=1`) for determinism and container-overshoot safety;
the offline `build_index.py` sets `EMBED_THREADS=12` (overrideable) to
parallelize the bulk re-embed.
