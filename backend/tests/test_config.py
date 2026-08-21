"""Regression tests for settings path handling."""

from pathlib import Path

from app.config import Settings, _REPO_ROOT


def test_relative_vector_store_path_anchors_to_repo_root():
    # .env carries './backend/data/index'. Resolved against CWD it silently
    # broke every query when the server ran from backend/ (chroma created an
    # empty store at backend/backend/data/index). Must resolve to repo root
    # regardless of process cwd.
    s = Settings(vector_store_path="./backend/data/index")
    assert s.vector_store_path == str(_REPO_ROOT / "backend" / "data" / "index")


def test_absolute_vector_store_path_untouched(tmp_path: Path):
    s = Settings(vector_store_path=str(tmp_path / "idx"))
    assert s.vector_store_path == str(tmp_path / "idx")
