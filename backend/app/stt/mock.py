"""Offline mock STT — for development without API keys.

NEVER let a TranscriptResult from this class reach a demo as if it were real:
is_mock is always True, the text says so inline, and constructing this
provider logs a warning. The factory only produces it when the configured
provider has no key or when STT_PROVIDER=mock is set explicitly.
"""

import logging

from app.stt.base import TranscriptResult

logger = logging.getLogger(__name__)

MOCK_TRANSCRIPT = "the quick brown fox jumps over the lazy dog"


class MockSTTProvider:
    def __init__(self) -> None:
        logger.warning(
            "STT running in MOCK mode — transcripts are canned test text, "
            "not real speech recognition. Do not demo this as real."
        )

    def transcribe(self, audio_bytes: bytes, mime_type: str) -> TranscriptResult:
        if not audio_bytes:
            # even the mock shouldn't pretend an empty upload transcribed fine
            raise ValueError("empty audio upload")
        return TranscriptResult(
            text=f"[mock] {MOCK_TRANSCRIPT}",
            provider="mock",
            is_mock=True,
            language="en",
        )
