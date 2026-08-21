#!/usr/bin/env python
"""Export ONE chunking strategy's collection as a standalone Chroma store.

Why: chromadb persists all collections into one shared chroma.sqlite3, so
the live index (4 strategies ≈ 186MB) can't be shipped per-collection by
copying files. The Docker build needs a small self-contained snapshot of the
query-time default strategy only — this script produces it.

Output: backend/data/index_snapshot.tgz — committed to the repo on purpose
(gitignored everything else under backend/data/) so the container image can
extract it at BUILD time and boot straight into warmup with zero runtime
network dependencies (no HF download, no re-embedding on deploy).

Usage:
    .venv/bin/python scripts/export_index_snapshot.py [--strategy metadata_aware]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings
from app.retrieval.snapshot import build_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = REPO_ROOT / "backend" / "data" / "index_snapshot.tgz"
MAX_SNAPSHOT_MB = 90  # GitHub hard-fails >100MB and warns >50MB


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default=get_settings().default_chunk_strategy)
    args = ap.parse_args()

    settings = get_settings()
    try:
        out = build_snapshot(args.strategy, settings, SNAPSHOT_PATH)
    except Exception as exc:
        raise SystemExit(
            f"snapshot failed: {exc} — is the index built? run scripts/build_index.py first"
        )

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"snapshot written: {out} ({size_mb:.1f} MB)")
    if size_mb > MAX_SNAPSHOT_MB:
        raise SystemExit(
            f"snapshot {size_mb:.1f}MB exceeds the {MAX_SNAPSHOT_MB}MB git-safety "
            "limit — trim the corpus or switch to Git LFS before committing"
        )


if __name__ == "__main__":
    main()
