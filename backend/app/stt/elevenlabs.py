"""ElevenLabs speech-to-text (Scribe models).

API shape verified against docs.elevenlabs.io (Aug 2026):
POST https://api.elevenlabs.io/v1/speech-to-text
  header  xi-api-key
  form    file, model_id (scribe_v1)
  returns {"text", "language_code", "language_probability", "audio_duration_secs"}
"""

from app.stt.base import TranscriptResult, post_multipart

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"


class ElevenLabsSTT:
    def __init__(
        self,
        api_key: str,
        model_id: str = "scribe_v1",
        timeout_seconds: float = 10.0,
        transport=None,
    ) -> None:
        if not api_key:
            raise ValueError("ElevenLabsSTT requires an API key")
        self.api_key = api_key
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self._transport = transport  # injectable for tests; None in production

    def transcribe(self, audio_bytes: bytes, mime_type: str) -> TranscriptResult:
        payload = post_multipart(
            url=ELEVENLABS_URL,
            headers={"xi-api-key": self.api_key},
            file_field="file",
            filename="audio",
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            form_fields={"model_id": self.model_id},
            timeout_seconds=self.timeout_seconds,
            transport=self._transport,
        )
        return TranscriptResult(
            text=payload.get("text", ""),
            provider="elevenlabs",
            language=payload.get("language_code"),
            # provider-reported probability; scribe gives no per-token
            # transcript confidence, so this is the closest honest signal
            confidence=payload.get("language_probability"),
            audio_duration_secs=payload.get("audio_duration_secs"),
        )
