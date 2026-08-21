"""Load real benchmark queries from the cached MSMARCO-XI shard.

The Eng_Query column holds genuine user questions, so benchmarks exercise
the pipeline with inputs shaped like production traffic rather than
hand-invented strings. Deduped and spread across the corpus (not just the
first N rows) so one hot region of the dataset can't dominate the sample.

Network quirk: reading via HfFileSystem hangs on this box — read the
already-downloaded parquet directly from the HF cache instead.
"""

from pathlib import Path

import pyarrow.parquet as pq

from app.config import Settings, get_settings


def load_real_queries(n: int, settings: Settings | None = None) -> list[str]:
    """n distinct real questions from the dataset's Eng_Query column."""
    if n <= 0:
        return []
    settings = settings or get_settings()
    filename = (
        f"{settings.dataset_language}"
        f"{'train' if settings.dataset_split == 'train' else 'val'}.parquet"
    )
    local = Path.home() / ".cache" / "huggingface" / "hub" / (
        "datasets--ai4bharat--MSMARCO-XI/snapshots"
    )
    shards = sorted(local.rglob(filename))
    if not shards:
        raise FileNotFoundError(
            "cached dataset shard not found — run scripts/download_and_prepare.py"
        )
    table = pq.read_table(str(shards[0]), columns=["Eng_Query"])
    queries = [q for q in table.column("Eng_Query").to_pylist() if q and len(q.strip()) > 10]
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    step = max(1, len(unique) // n)
    picked = [unique[i] for i in range(0, len(unique), step)][:n]
    if len(picked) < n:
        raise ValueError(f"only {len(picked)} distinct queries available, wanted {n}")
    return picked
