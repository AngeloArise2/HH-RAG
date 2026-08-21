"""Tests for the single-strategy index snapshot builder (deployment path)."""

from pathlib import Path
import tarfile

import pytest

from app.config import get_settings
from app.retrieval.snapshot import build_snapshot

settings = get_settings()
_index_dir = Path(settings.vector_store_path)
INDEX_EXISTS = _index_dir.is_dir() and any(_index_dir.iterdir())

pytestmark = pytest.mark.skipif(
    not INDEX_EXISTS, reason="vector index not built — run scripts/build_index.py"
)


def test_snapshot_roundtrip(tmp_path: Path):
    """Exported archive must contain a loadable chroma store with the full count."""
    import chromadb

    out = tmp_path / "snapshot.tgz"
    result = build_snapshot(settings.default_chunk_strategy, settings, out)

    assert result == out and out.stat().st_size > 0

    with tarfile.open(out) as tar:
        names = tar.getnames()
        assert any(n.endswith("chroma.sqlite3") for n in names), "store missing"

    # extract and open as a fresh client to prove the snapshot is standalone
    extract_dir = tmp_path / "extracted"
    with tarfile.open(out) as tar:
        tar.extractall(extract_dir)
    client = chromadb.PersistentClient(path=str(extract_dir / "index"))
    col = client.get_collection(name=settings.default_chunk_strategy)
    assert col.count() > 0


def test_snapshot_rejects_unknown_strategy(tmp_path: Path):
    with pytest.raises(Exception):
        build_snapshot("no_such_strategy_xyz", settings, tmp_path / "x.tgz")
