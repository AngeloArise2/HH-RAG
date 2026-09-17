#!/usr/bin/env python
"""Build one Chroma collection per chunking strategy from real ingested data.

Pipeline: processed/passages.jsonl -> each strategy's chunker -> embed all
chunks (local English-only MiniLM) -> persist collection under VECTOR_STORE_PATH.

Reports honest wall-clock timing per strategy. Formal per-stage latency
instrumentation (stage_timer/LatencyTrace) lands in phase 4; this script's
numbers are simple wall-clock around the full embed+index stage, not fudged
into anything else.

Usage: .venv/bin/python scripts/build_index.py
"""

import os
import sys
import time
from pathlib import Path

# Offline bulk embed: use parallelism (the runtime hot path stays single-
# threaded; only this offline build scales up ORT threads). An existing value
# in the environment (EMBED_THREADS=...) always wins.
os.environ.setdefault("EMBED_THREADS", "12")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.chunking import router
from app.config import get_settings
from app.ingestion.models import RawDocument
from app.retrieval import vector_store


def load_documents() -> list[RawDocument]:
    path = get_settings().processed_data_dir / "passages.jsonl"
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(RawDocument.model_validate_json(line))
    if not docs:
        raise SystemExit(f"no documents in {path} — run scripts/download_and_prepare.py")
    return docs


def main() -> None:
    settings = get_settings()
    docs = load_documents()
    print(f"corpus: {len(docs)} processed passages\n")

    total_start = time.perf_counter()
    grand_total_chunks = 0
    results = []
    for name, chunker in router.STRATEGIES.items():
        chunks = [c for d in docs for c in chunker.chunk(d)]
        start = time.perf_counter()
        n_indexed = vector_store.build_index(chunks, name, settings=settings)
        elapsed = time.perf_counter() - start
        grand_total_chunks += n_indexed
        rate = n_indexed / elapsed if elapsed else 0
        results.append((name, n_indexed, elapsed, rate))
        print(f"[{name}] {n_indexed} chunks indexed in {elapsed:.1f}s ({rate:.0f} chunks/s)")

    total_elapsed = time.perf_counter() - total_start
    print(f"\ntotal: {grand_total_chunks} chunks across {len(results)} collections "
          f"in {total_elapsed:.1f}s")

    print("\ncollection sizes on disk:")
    for name, count in vector_store.collection_counts(settings).items():
        print(f"  {name}: {count} vectors")


if __name__ == "__main__":
    main()
