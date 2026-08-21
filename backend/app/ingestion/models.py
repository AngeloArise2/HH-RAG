"""Shared document models passed between ingestion and chunking."""

from typing import Any

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    """A single standalone passage flowing from ingestion into chunking.

    `id` is assigned during preprocessing (sha1[:16] of the cleaned text) so it
    is stable across re-runs regardless of row order; freshly downloaded raw
    documents carry `id=None` until preprocessing stamps them.
    """

    id: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
