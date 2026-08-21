"""Download a bounded working subset of ai4bharat/MSMARCO-XI.

The full dataset is ~55GB spread over per-language parquet shards (~4GB each
for train, ~460MB for validation). We pull exactly ONE shard via
huggingface_hub's resumable download and read only the first `max_raw_rows`
query rows out of it with pyarrow, so the cost is one bounded file download,
not the whole dataset.

Each row is one query with NESTED passage lists (English_passages /
Translated_passages / is_selected), not a flat text corpus. We flatten only
English_passages into standalone RawDocuments — one row can yield multiple —
preserving query_id / query_type / is_selected in metadata. See the "Dataset
schema" section of docs/architecture.md for why.
"""

from collections.abc import Iterator
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from app.config import Settings, get_settings
from app.ingestion.models import RawDocument

RAW_FILENAME = "passages_raw.jsonl"
SOURCE_DATASET = "ai4bharat/MSMARCO-XI"


def shard_filename(split: str, language: str) -> str:
    """Map (split, language) to the shard layout used in the HF repo."""
    suffix = "train" if split == "train" else "val"
    return f"{split}/{language}{suffix}.parquet"


def row_to_documents(row: dict[str, Any], *, language: str, split: str) -> list[RawDocument]:
    """Flatten one query row into standalone passage documents.

    Only English_passages become documents; Translated_passages are ignored.
    is_selected is index-aligned with English_passages — if the flag list is
    shorter or missing we fall back to False rather than mislabeling.
    """
    passages = row.get("passages") or {}
    texts = passages.get("English_passages") or []
    selected = passages.get("is_selected") or []
    return [
        RawDocument(
            text=text,
            metadata={
                "source_dataset": SOURCE_DATASET,
                "language": language,
                "split": split,
                "query_id": row.get("query_id"),
                "query_type": row.get("query_type"),
                "is_selected": bool(selected[i]) if i < len(selected) else False,
                "passage_index": i,
            },
        )
        for i, text in enumerate(texts)
    ]


def iter_row_documents(settings: Settings) -> Iterator[list[RawDocument]]:
    """Yield one list of RawDocuments per query row, up to max_raw_rows rows.

    Reading happens lazily via pyarrow iter_batches so memory stays flat no
    matter how large the shard is; download itself is cached+resumable by
    huggingface_hub, so an interrupted run continues instead of restarting.
    """
    local_path = hf_hub_download(
        repo_id=settings.dataset_name,
        filename=shard_filename(settings.dataset_split, settings.dataset_language),
        repo_type="dataset",
    )
    parquet_file = pq.ParquetFile(local_path)
    rows_read = 0
    for batch in parquet_file.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            yield row_to_documents(
                row, language=settings.dataset_language, split=settings.dataset_split
            )
            rows_read += 1
            if rows_read >= settings.max_raw_rows:
                return


def run(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    raw_dir = settings.raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / RAW_FILENAME

    query_rows = 0
    passages_written = 0
    total_chars = 0
    samples: list[RawDocument] = []

    with out_path.open("w", encoding="utf-8") as f:
        for docs in iter_row_documents(settings):
            query_rows += 1
            for doc in docs:
                f.write(doc.model_dump_json() + "\n")
                passages_written += 1
                total_chars += len(doc.text)
                if len(samples) < 2:
                    samples.append(doc)

    avg_chars = total_chars / passages_written if passages_written else 0.0
    stats: dict[str, Any] = {
        "query_rows": query_rows,
        "passages_written": passages_written,
        "avg_passage_chars": round(avg_chars, 1),
        "output_file": str(out_path),
        "samples": [
            {
                "query_id": s.metadata["query_id"],
                "query_type": s.metadata["query_type"],
                "is_selected": s.metadata["is_selected"],
                "text_preview": s.text[:120],
            }
            for s in samples
        ],
    }

    print(f"[download] query rows read:      {query_rows}")
    print(f"[download] passages written:     {passages_written}")
    print(f"[download] avg passage length:   {avg_chars:.0f} chars")
    print(f"[download] output:               {out_path}")
    for i, s in enumerate(stats["samples"]):
        print(f"[download] sample {i + 1}: query_id={s['query_id']} type={s['query_type']} "
              f"selected={s['is_selected']}\n           {s['text_preview']!r}")
    return stats


if __name__ == "__main__":
    run(get_settings())
