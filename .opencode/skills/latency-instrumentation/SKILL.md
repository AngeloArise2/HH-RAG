---
name: latency-instrumentation
description: Playbook for adding timing instrumentation and computing P50/P70/P100 latency stats for this project's RAG pipeline. Use whenever touching backend/app/benchmarking/, adding a new pipeline stage that should be timed, or writing/updating scripts/run_benchmark.py.
license: MIT
compatibility: opencode
---

## What I do

Keep latency measurement centralized and honest across the whole pipeline instead of ad-hoc `time.time()` calls scattered through the codebase.

## The timer utility

`backend/app/benchmarking/latency.py` exposes a context manager:

```python
from app.benchmarking.latency import stage_timer

with stage_timer("chunk_lookup") as t:
    ...
with stage_timer("vector_search") as t:
    ...
```

Every stage's duration gets recorded into a `LatencyTrace` object attached to the request, with named stages: `stt`, `embed_query`, `vector_search`, `chunk_assembly`, `generation`, `guardrail_check`. Any new pipeline stage must use this same mechanism — don't add a parallel timing system.

## Computing percentiles

`scripts/run_benchmark.py` runs N queries (default 50, configurable), collects a `LatencyTrace` per query, and computes P50/P70/P100 two ways:
1. **Retrieval-only** = `embed_query + vector_search + chunk_assembly` (this is what's checked against the 200ms target)
2. **Full end-to-end** = all stages including `stt` and `generation` (reported honestly, no target claimed)

Output goes to `docs/latency_report.md` as a table plus a one-paragraph honest summary — never silently exclude a slow stage from the "full end-to-end" number to make it look better.

## Rule for any new external call (API, network, disk)

If you add a call to STT, an LLM, or any network service, wrap it in `stage_timer` immediately in the same commit — don't ship untimed I/O and plan to "add benchmarking later."
