"""Clean raw downloaded passages into processed/passages.jsonl.

Steps: strip HTML artifacts and collapsed whitespace, drop empty/near-empty
passages, dedupe by content, and assign a stable doc_id (sha1[:16] of the
cleaned text — content-derived so ids survive re-runs even if row order or
the source shard changes). Duplicate passages keep their first occurrence but
OR-merge the is_selected flag so a passage selected under any query stays
marked for future evaluation use.
"""

import hashlib
import html
import json
import re
from collections.abc import Iterable
from typing import Any

from app.config import Settings, get_settings
from app.ingestion.download_dataset import RAW_FILENAME
from app.ingestion.models import RawDocument

PROCESSED_FILENAME = "passages.jsonl"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip HTML tags/entities and collapse whitespace runs."""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def stable_doc_id(cleaned_text: str) -> str:
    """Content-derived id, stable across runs and independent of row order."""
    return hashlib.sha1(cleaned_text.encode("utf-8")).hexdigest()[:16]


def preprocess_documents(
    docs: Iterable[RawDocument], *, min_chars: int
) -> tuple[list[RawDocument], dict[str, int]]:
    """Clean + filter + dedupe; returns (kept_documents, stats)."""
    seen: dict[str, RawDocument] = {}
    kept: list[RawDocument] = []
    dropped_empty = dropped_short = duplicates = 0

    for doc in docs:
        cleaned = clean_text(doc.text)
        if not cleaned:
            dropped_empty += 1
            continue
        if len(cleaned) < min_chars:
            dropped_short += 1
            continue
        doc_id = stable_doc_id(cleaned)
        if doc_id in seen:
            duplicates += 1
            prev = seen[doc_id]
            prev.metadata["is_selected"] = bool(
                prev.metadata.get("is_selected") or doc.metadata.get("is_selected", False)
            )
            continue
        merged = doc.model_copy(update={"id": doc_id, "text": cleaned})
        seen[doc_id] = merged
        kept.append(merged)

    stats = {
        "input": dropped_empty + dropped_short + duplicates + len(kept),
        "dropped_empty": dropped_empty,
        "dropped_near_empty": dropped_short,
        "duplicates_merged": duplicates,
        "kept": len(kept),
    }
    return kept, stats


def run(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    in_path = settings.raw_data_dir / RAW_FILENAME
    out_path = settings.processed_data_dir / PROCESSED_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open(encoding="utf-8") as f:
        docs = (RawDocument.model_validate_json(line) for line in f if line.strip())
        kept, stats = preprocess_documents(docs, min_chars=settings.min_passage_chars)

    with out_path.open("w", encoding="utf-8") as f:
        for doc in kept:
            f.write(doc.model_dump_json() + "\n")

    lengths = [len(d.text) for d in kept]
    avg_chars = sum(lengths) / len(lengths) if lengths else 0.0
    stats.update(
        {
            "min_passage_chars": min(lengths, default=0),
            "max_passage_chars": max(lengths, default=0),
            "avg_passage_chars": round(avg_chars, 1),
            "output_file": str(out_path),
            "samples": [
                {"id": d.id, "text_preview": d.text[:120]} for d in kept[:2]
            ],
        }
    )

    print(f"[preprocess] input passages:      {stats['input']}")
    print(f"[preprocess] dropped empty:       {stats['dropped_empty']}")
    print(f"[preprocess] dropped near-empty:  {stats['dropped_near_empty']} "
          f"(< {settings.min_passage_chars} chars)")
    print(f"[preprocess] duplicates merged:   {stats['duplicates_merged']}")
    print(f"[preprocess] final passages:      {stats['kept']}")
    print(f"[preprocess] length min/avg/max:  {stats['min_passage_chars']}/"
          f"{stats['avg_passage_chars']}/{stats['max_passage_chars']} chars")
    print(f"[preprocess] output:              {out_path}")
    for i, s in enumerate(stats["samples"]):
        print(f"[preprocess] sample {i + 1}: id={s['id']}\n             {s['text_preview']!r}")
    return stats


if __name__ == "__main__":
    run(get_settings())
