"""Provider factory: config decides which STT implementation serves requests.

Policy:
- STT_PROVIDER=sarvam|elevenlabs with a key present -> that real provider.
- Real provider named but its key is EMPTY -> fall back to the mock with a
  loud warning. Dev keeps working; the response's is_mock flag (and the
  mock_note on /transcribe) make it impossible to mistake for real output.
- STT_PROVIDER=mock -> mock, no warnings about missing keys.
- Unknown provider NAME -> hard error: a typo in config must not be masked.
"""

import logging

from app.config import Settings, get_settings
from app.stt.base import STTProvider
from app.stt.elevenlabs import ElevenLabsSTT
from app.stt.mock import MockSTTProvider
from app.stt.sarvam import SarvamSTT

logger = logging.getLogger(__name__)


def get_stt_provider(settings: Settings | None = None) -> STTProvider:
    settings = settings or get_settings()
    name = settings.stt_provider.strip().lower()

    if name == "mock":
        return MockSTTProvider()
    if name == "sarvam":
        if settings.sarvam_api_key:
            return SarvamSTT(api_key=settings.sarvam_api_key)
        logger.warning("STT_PROVIDER=sarvam but SARVAM_API_KEY is empty — using MOCK STT")
        return MockSTTProvider()
    if name == "elevenlabs":
        if settings.elevenlabs_api_key:
            return ElevenLabsSTT(api_key=settings.elevenlabs_api_key)
        logger.warning(
            "STT_PROVIDER=elevenlabs but ELEVENLABS_API_KEY is empty — using MOCK STT"
        )
        return MockSTTProvider()

    raise ValueError(
        f"unknown STT_PROVIDER {settings.stt_provider!r} (expected sarvam/elevenlabs/mock)"
    )
