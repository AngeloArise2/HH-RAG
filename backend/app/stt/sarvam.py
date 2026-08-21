"""Sarvam AI speech-to-text (Saaras/Saarika models).

API shape verified against docs.sarvam.ai (Aug 2026):
POST https://api.sarvam.ai/speech-to-text
  header  api-subscription-key
  form    file (wav/webm/mp3/ogg/flac), model, language_code
  returns {"request_id", "transcript", "language_code", ...}
Sync endpoint accepts up to 30s of audio.
"""

import logging

from app.stt.base import TranscriptResult, post_multipart

logger = logging.getLogger(__name__)

SARVAM_URL = "https://api.sarvam.ai/speech-to-text"


class SarvamSTT:
    def __init__(
        self,
        api_key: str,
        model: str = "saaras:v3",
        # live API (Aug 2026) accepts 'unknown' for auto-detect; 'auto' is
        # rejected with a 400 listing the full locale enum
        language_code: str = "unknown",
        timeout_seconds: float = 10.0,
        transport=None,
    ) -> None:
        if not api_key:
            raise ValueError("SarvamSTT requires an API key")
        self.api_key = api_key
        self.model = model
        self.language_code = language_code
        self.timeout_seconds = timeout_seconds
        self._transport = transport  # injectable for tests; None in production

    def transcribe(self, audio_bytes: bytes, mime_type: str) -> TranscriptResult:
        payload = post_multipart(
            url=SARVAM_URL,
            headers={"api-subscription-key": self.api_key},
            file_field="file",
            filename="audio",
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            form_fields={"model": self.model, "language_code": self.language_code},
            timeout_seconds=self.timeout_seconds,
            transport=self._transport,
        )
        logger.debug("sarvam stt request_id=%s", payload.get("request_id"))
        return TranscriptResult(
            text=payload.get("transcript", ""),
            provider="sarvam",
            language=payload.get("language_code"),
        )
