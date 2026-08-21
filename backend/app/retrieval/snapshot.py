"""Snapshot one strategy's collection into a standalone Chroma store archive.

chromadb persists ALL collections into one shared chroma.sqlite3, so a
per-collection subset can't be shipped by copying files. Deployment needs a
small self-contained snapshot of only the query-time default strategy — this
module produces it. See scripts/export_index_snapshot.py for the CLI.
"""

import tarfile
import tempfile
from pathlib import Path

import chromadb

from app.config import Settings


def _copy_collection(source_dir: str | Path, strategy_name: str, target_dir: Path) -> int:
    """Copy one collection into a fresh PersistentClient dir; returns count."""
    src = chromadb.PersistentClient(path=str(source_dir))
    col = src.get_collection(name=strategy_name)
    count = col.count()
    if count == 0:
        raise ValueError(f"collection {strategy_name!r} is empty — nothing to export")

    rows = col.get(include=["documents", "metadatas", "embeddings"])

    dst = chromadb.PersistentClient(path=str(target_dir))
    new_col = dst.get_or_create_collection(
        name=strategy_name, metadata={"hnsw:space": "cosine"}
    )
    batch = 2000  # chroma per-call cap safety margin
    ids = rows["ids"]
    for i in range(0, len(ids), batch):
        sl = slice(i, i + batch)
        new_col.upsert(
            ids=ids[sl],
            embeddings=rows["embeddings"][sl],
            documents=rows["documents"][sl],
            metadatas=[dict(m or {}) for m in rows["metadatas"][sl]],
        )
    exported = new_col.count()
    if exported != count:
        raise RuntimeError(f"snapshot mismatch: wrote {exported}, expected {count}")
    return exported


def build_snapshot(
    strategy_name: str,
    settings: Settings,
    out_path: Path,
) -> Path:
    """Archive one strategy as <out_path> tgz containing a top-level index/."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "index"
        _copy_collection(settings.vector_store_path, strategy_name, staged)
        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(str(staged), arcname="index")
    return out_path
