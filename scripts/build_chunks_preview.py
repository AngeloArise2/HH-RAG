#!/usr/bin/env python
"""Run all four chunking strategies against a small real sample from
backend/data/processed/passages.jsonl and print per-strategy stats plus one
example chunk each — so the behavioral differences are visible by eye.

Usage: .venv/bin/python scripts/build_chunks_preview.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.chunking import router
from app.config import get_settings
from app.ingestion.models import RawDocument

N_DOCS = 25


def load_sample(n: int) -> list[RawDocument]:
    path = get_settings().processed_data_dir / "passages.jsonl"
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if len(docs) >= n:
                break
            if line.strip():
                docs.append(RawDocument.model_validate_json(line))
    if not docs:
        raise SystemExit(f"no documents found in {path} — run scripts/download_and_prepare.py")
    return docs


def main() -> None:
    docs = load_sample(N_DOCS)
    total_words = sum(len(d.text.split()) for d in docs)
    print(f"sample: {len(docs)} real passages, {total_words} words total")
    print("note: MSMARCO passages are short (~47 words avg), so demo configs below\n"
          "use small windows that actually exercise each strategy's splitting logic;\n"
          "defaults live in the chunker constructors.\n")

    # Configs chosen so real passages actually split into multiple chunks.
    demo_configs = {
        "fixed_size": lambda: router.FixedSizeChunker(max_words=30),
        "fixed_size_overlap": lambda: router.FixedSizeOverlapChunker(
            max_words=30, overlap_ratio=1 / 3
        ),
        "semantic": lambda: router.SemanticChunker(max_words=45),
        "metadata_aware": lambda: router.MetadataAwareChunker(
            inner=router.SemanticChunker(max_words=45)
        ),
    }

    for name, factory in demo_configs.items():
        chunker = factory()
        all_chunks = [c for d in docs for c in chunker.chunk(d)]
        word_counts = [len(c.text.split()) for c in all_chunks]
        avg = sum(word_counts) / len(word_counts) if word_counts else 0
        multi = sum(1 for d in docs if len(chunker.chunk(d)) > 1)
        print(f"=== {name} ===")
        print(f"chunks: {len(all_chunks)}  |  docs split into >1 chunk: {multi}/{len(docs)}"
              f"  |  words/chunk min/avg/max: {min(word_counts)}/{avg:.1f}/{max(word_counts)}")

        example = max(all_chunks, key=lambda c: len(c.text))  # most representative-ish
        preview = example.text if len(example.text) <= 220 else example.text[:220] + "…"
        print(f"example [{example.chunk_id}] offset {example.start_offset}-{example.end_offset}:")
        print(f"  {preview!r}")
        if name == "metadata_aware":
            print(f"  metadata keys: {sorted(example.metadata.keys())}")
            print(f"  doc_query_id={example.metadata.get('doc_query_id')} "
                  f"doc_is_selected={example.metadata.get('doc_is_selected')} "
                  f"inner_strategy={example.metadata.get('inner_strategy')}")
        print()


if __name__ == "__main__":
    main()
