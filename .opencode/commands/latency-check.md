---
description: Run the latency benchmark and report P50/P70/P100 against the 200ms retrieval target
agent: build
---

Run the benchmark script:

!`.venv/bin/python scripts/run_benchmark.py --queries 50 --sleep 15 --output docs/latency_report.md 2>&1 | tail -60`

Read the output above and @docs/latency_report.md if it was written. Report:

1. P50 / P70 / P100 for the **retrieval-only** path (chunking + vector search) — state clearly whether this meets the sub-200ms target.
2. P50 / P70 / P100 for the **full end-to-end** path (including STT and LLM generation) — reported honestly, no target claimed against it, just the real numbers.
3. Anything that looks like an outlier or a broken measurement (e.g. a suspiciously flat 0ms, or one query 100x slower than the rest) — call it out, don't silently average it away.
4. If the benchmark script doesn't exist yet or fails, say so plainly and point at which phase in `BUILD_PROMPT.md` builds it (Phase 8) — don't improvise a fake substitute measurement.

Argument (optional — number of test queries to run, default 50): $ARGUMENTS
