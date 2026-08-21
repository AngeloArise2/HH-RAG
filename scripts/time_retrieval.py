#!/usr/bin/env python
"""Sanity-check retrieval latency: run 20 real queries through the Retriever
and print per-stage millisecond timings + a rough P50.

Real queries come from the dataset's Eng_Query column (read from the already-
cached validation shard), so we time the kind of inputs the pipeline will
actually see. Full percentile machinery (P50/P70/P100, N>=50, report file)
lands in phase 8's scripts/run_benchmark.py — this is a quick honest signal.

Usage: .venv/bin/python scripts/time_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.benchmarking.queries import load_real_queries
from app.benchmarking.latency import RETRIEVAL_STAGES
from app.config import get_settings
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import collection_counts

N_QUERIES = 20


def main() -> None:
    settings = get_settings()
    counts = collection_counts(settings)
    if counts.get(settings.default_chunk_strategy, 0) == 0:
        raise SystemExit(
            f"no index for strategy {settings.default_chunk_strategy!r} "
            f"(have: {counts}) — run scripts/build_index.py"
        )
    print(f"index: {counts[settings.default_chunk_strategy]} vectors in "
          f"'{settings.default_chunk_strategy}'\n")

    retriever = Retriever(top_k=5)

    queries = load_real_queries(N_QUERIES)
    if len(queries) < N_QUERIES:
        raise SystemExit(f"only got {len(queries)} distinct real queries")

    # warm-up (model load + first ANN touch); reported but excluded from P50
    warm = retriever.retrieve(queries[0])
    print(f"warm-up (excluded): embed={warm.trace.stage_ms.get('embed_query')}ms "
          f"search={warm.trace.stage_ms.get('vector_search')}ms\n")

    print(f"{'#':>2} {'embed_query':>12} {'vector_search':>14} {'retrieval':>10}  query")
    embed_times, search_times, total_times = [], [], []
    for i, q in enumerate(queries):
        result = retriever.retrieve(q)
        e = result.trace.stage_ms.get("embed_query", 0.0)
        s = result.trace.stage_ms.get("vector_search", 0.0)
        r = result.trace.retrieval_ms()
        embed_times.append(e)
        search_times.append(s)
        total_times.append(r)
        print(f"{i:>2} {e:>9.1f}ms {s:>11.1f}ms {r:>7.1f}ms  {q[:48]!r}")

    def p50(xs):
        return sorted(xs)[len(xs) // 2]

    print(f"\nrough P50 over {N_QUERIES} queries (full percentiles land in phase 8):")
    print(f"  embed_query : {p50(embed_times):.1f}ms")
    print(f"  vector_search: {p50(search_times):.1f}ms")
    print(f"  retrieval    : {p50(total_times):.1f}ms  (target: <200ms)")
    verdict = "COMFORTABLY UNDER" if p50(total_times) < 100 else (
        "UNDER" if p50(total_times) < 200 else "OVER — needs attention NOW, not at phase 8")
    print(f"\nverdict: {verdict} the 200ms target")


if __name__ == "__main__":
    main()
