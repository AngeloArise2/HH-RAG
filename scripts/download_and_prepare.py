#!/usr/bin/env python
"""Run ingestion end-to-end: download a bounded MSMARCO-XI subset, then clean
it into backend/data/processed/passages.jsonl.

Usage: .venv/bin/python scripts/download_and_prepare.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings
from app.ingestion import download_dataset, preprocess


def main() -> None:
    settings = get_settings()
    print(f"=== download: {settings.dataset_name} "
          f"[{settings.dataset_split}/{settings.dataset_language}, "
          f"cap {settings.max_raw_rows} query rows] ===")
    download_stats = download_dataset.run(settings)
    if download_stats["passages_written"] == 0:
        raise SystemExit("download produced 0 passages; not running preprocess")

    print("\n=== preprocess ===")
    prep_stats = preprocess.run(settings)
    if prep_stats["kept"] == 0:
        raise SystemExit("preprocess kept 0 passages — inspect raw output")


if __name__ == "__main__":
    main()
