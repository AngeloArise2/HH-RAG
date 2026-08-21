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
