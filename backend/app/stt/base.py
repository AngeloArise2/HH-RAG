"""Speech-to-text contracts shared by every provider.

Request-path convention: callers pass raw audio bytes + MIME type; providers
return a TranscriptResult. The browser (phase 9) records MediaRecorder's
default audio/webm;codecs=opus — both providers accept webm directly, so no
conversion layer exists anywhere in this pipeline.

Every real network call must be wrapped in stage_timer("stt", trace) by the
CALLER (harness/endpoint), keeping providers timing-agnostic and unit-testable.
"""

from typing import Any, Optional, Protocol

import httpx
from pydantic import BaseModel
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class TranscriptResult(BaseModel):
    """One transcription outcome. is_mock=True means the text came from the
    offline mock provider — responses carrying it MUST surface that flag."""

    text: str
    provider: str
    is_mock: bool = False
    language: Optional[str] = None
    confidence: Optional[float] = None
    audio_duration_secs: Optional[float] = None


class STTError(RuntimeError):
    """Raised after retries are exhausted or the failure is permanent."""


class STTProvider(Protocol):
    def transcribe(self, audio_bytes: bytes, mime_type: str) -> TranscriptResult: ...


# --- Shared HTTP machinery: one retry/backoff/timeout path for all providers ---

_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3


class TransientStatusError(Exception):
    """HTTP status worth retrying (429 / 5xx)."""


RETRYABLE_EXCEPTIONS = (
    TransientStatusError,
    httpx.TimeoutException,
    httpx.TransportError,
)


def post_multipart(
    *,
    url: str,
    headers: dict[str, str],
    file_field: str,
    filename: str,
    audio_bytes: bytes,
    mime_type: str,
    form_fields: dict[str, str],
    timeout_seconds: float,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """POST multipart with retry + exponential backoff on transient failures.

    - Retries timeouts, transport errors, and 429/5xx up to 3 attempts,
      waiting 0.5s -> 1s between tries.
    - Permanent failures (401 bad key, 413 too large, 422 bad format, ...)
      fail immediately on the first attempt — retrying those is pointless.
    - Raises STTError once retries are exhausted or the request is dead;
      callers never see raw httpx exceptions.
    """

    def send_once() -> dict[str, Any]:
        if not audio_bytes:
            raise ValueError("empty audio payload")
        with httpx.Client(timeout=timeout_seconds, transport=transport) as client:
            response = client.post(
                url,
                headers=headers,
                files={file_field: (filename, audio_bytes, mime_type)},
                data=form_fields,
            )
            if response.status_code >= 400:
                if response.status_code in _TRANSIENT_STATUS:
                    raise TransientStatusError(
                        f"{response.status_code}: {response.text[:200]}"
                    )
                raise STTError(
                    f"STT API returned {response.status_code}: {response.text[:200]}"
                )
            try:
                return response.json()
            except ValueError as exc:
                # a 200 with a non-JSON body is still a failed transcription
                raise STTError(
                    f"STT API returned non-JSON body: {response.text[:200]}"
                ) from exc

    try:
        for attempt in Retrying(
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            stop=stop_after_attempt(_MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
            reraise=True,
        ):
            with attempt:
                return send_once()
        raise AssertionError("unreachable: Retrying always returns or raises")
    except RETRYABLE_EXCEPTIONS as exc:
        raise STTError(f"STT API unreachable after {_MAX_ATTEMPTS} attempts: {exc}") from exc
