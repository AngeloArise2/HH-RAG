---
name: chunking-strategy
description: Playbook for implementing and extending multi-strategy document chunking for this project's RAG pipeline (fixed-size, overlap, semantic, metadata-aware). Use whenever adding, modifying, or reviewing a chunker in backend/app/chunking/.
license: MIT
compatibility: opencode
---

## What I do

Give consistent rules for any chunking work in this repo, so every strategy is comparable and swappable behind one interface.

## Interface every chunker must implement

```python
class Chunk(BaseModel):
    text: str
    doc_id: str
    chunk_id: str
    start_offset: int
    end_offset: int
    metadata: dict  # source title, section, strategy name, position index

class Chunker(Protocol):
    def chunk(self, document: RawDocument) -> list[Chunk]: ...
```

All chunkers live in `backend/app/chunking/`, one file per strategy, registered in `router.py`'s `STRATEGIES` dict by name.

## Strategies to implement (minimum set — task explicitly forbids a single naive fixed-size approach)

1. **fixed_size.py** — fixed token/word count, no overlap. This is the baseline, not the answer.
2. **fixed_size_overlap.py** — same, but with a configurable overlap window (e.g. 20% overlap) so ideas spanning a boundary survive.
3. **semantic.py** — split on sentence boundaries, grouping sentences up to a soft max length rather than a hard cutoff mid-sentence.
4. **metadata_aware.py** — wraps another strategy but enriches every chunk's `metadata` with document-level fields (title, section, source row from the dataset) so retrieval can filter/boost on them later.

## Testing rule

Every chunker gets a test in `backend/tests/test_chunking_<name>.py` asserting:
- No empty chunks
- No chunk exceeds the configured max length
- (overlap strategies only) adjacent chunks actually share the configured overlap
- Chunk count is deterministic for the same input + config

## When comparing strategies

Don't just eyeball output — the benchmark script (Phase 8) should be able to run retrieval against an index built from each strategy and report retrieval quality/latency per strategy, so the "vast chunking" claim in the submission is backed by a comparison, not just four files that all exist unused.
