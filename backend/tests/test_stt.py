"""STT tests: factory policy, retry/backoff behavior, mock endpoint flow.

All network behavior runs through httpx.MockTransport — zero real HTTP in the
unit suite. The live-provider check lives in scripts/test_stt_real.py.
"""

import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, stt_provider_dependency
from app.stt.base import STTError, TranscriptResult, post_multipart
from app.stt.elevenlabs import ElevenLabsSTT
from app.stt.factory import get_stt_provider
from app.stt.mock import MockSTTProvider
from app.stt.sarvam import SarvamSTT

# --- fixtures/helpers -------------------------------------------------------

PNG_BYTES = b"\x00\x01\x02fake-audio-payload"


def transport_with_status(statuses: list[int], final_json: dict | None = None):
    """MockTransport returning given statuses in order, then final_json."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        n = calls["count"]
        calls["count"] += 1
        if n >= len(statuses):  # list exhausted -> real success
            return httpx.Response(200, json=final_json or {"transcript": "ok"})
        status = statuses[n]
        if status == 200:
            return httpx.Response(200, json=final_json or {"transcript": "ok"})
        return httpx.Response(status, text=f"status {status}")

    return httpx.MockTransport(handler), calls


# --- factory policy ---------------------------------------------------------

def test_factory_returns_sarvam_when_key_present():
    settings = Settings(stt_provider="sarvam", sarvam_api_key="k-test")
    provider = get_stt_provider(settings)
    assert isinstance(provider, SarvamSTT)


def test_factory_falls_back_to_mock_without_key(caplog):
    settings = Settings(stt_provider="sarvam", sarvam_api_key="")
    with caplog.at_level(logging.WARNING):
        provider = get_stt_provider(settings)
    assert isinstance(provider, MockSTTProvider)
    assert any("MOCK" in r.message for r in caplog.records)


def test_factory_elevenlabs_fallback_without_key():
    settings = Settings(stt_provider="elevenlabs", elevenlabs_api_key="")
    assert isinstance(get_stt_provider(settings), MockSTTProvider)


def test_factory_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown STT_PROVIDER"):
        get_stt_provider(Settings(stt_provider="whisper"))


def test_explicit_mock_mode_is_silent_choice():
    assert isinstance(get_stt_provider(Settings(stt_provider="mock")), MockSTTProvider)


def test_providers_require_keys():
    with pytest.raises(ValueError):
        SarvamSTT(api_key="")
    with pytest.raises(ValueError):
        ElevenLabsSTT(api_key="")


# --- retry / timeout / error handling (the "demonstrably present" part) -----

def test_transient_500s_are_retried_then_success():
    transport, calls = transport_with_status(
        [503, 500],
        final_json={"transcript": "recovered transcript", "language_code": "en-IN"},
    )
    provider = SarvamSTT(api_key="k", timeout_seconds=5.0, transport=transport)
    result = provider.transcribe(PNG_BYTES, "audio/webm")
    assert calls["count"] == 3  # two transient failures + one success
    assert result.text == "recovered transcript"
    assert result.provider == "sarvam"


def test_rate_limit_429_retried_then_success():
    transport, calls = transport_with_status([429], final_json={"transcript": "ok"})
    provider = SarvamSTT(api_key="k", transport=transport)
    provider.transcribe(PNG_BYTES, "audio/webm")
    assert calls["count"] >= 2


def test_retries_exhausted_raises_stterror():
    transport, calls = transport_with_status([503] * 10)
    provider = SarvamSTT(api_key="k", timeout_seconds=0.2, transport=transport)
    with pytest.raises(STTError, match="unreachable after 3 attempts"):
        provider.transcribe(PNG_BYTES, "audio/webm")
    assert calls["count"] == 3


def test_timeout_exception_is_retryable():
    def handler(request):
        raise httpx.ConnectTimeout("hung")

    provider = SarvamSTT(
        api_key="k",
        timeout_seconds=0.1,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(STTError, match="unreachable"):
        provider.transcribe(PNG_BYTES, "audio/webm")


def test_permanent_403_fails_fast_without_retry():
    transport, calls = transport_with_status([403])
    provider = SarvamSTT(api_key="bad-key", transport=transport)
    with pytest.raises(STTError, match="403"):
        provider.transcribe(PNG_BYTES, "audio/webm")
    assert calls["count"] == 1  # no pointless retries on a bad key


def test_elevenlabs_parses_scribe_response():
    transport, _ = transport_with_status(
        [200],
        final_json={
            "text": "hello from scribe",
            "language_code": "eng",
            "language_probability": 0.98,
            "audio_duration_secs": 1.5,
        },
    )
    provider = ElevenLabsSTT(api_key="k", transport=transport)
    result = provider.transcribe(PNG_BYTES, "audio/webm")
    assert isinstance(result, TranscriptResult)
    assert result.text == "hello from scribe"
    assert result.language == "eng"
    assert result.confidence == 0.98
    assert result.audio_duration_secs == 1.5


def test_malformed_json_body_raises_stterror_not_crash():
    def handler(request):
        return httpx.Response(200, text="<html>gateway garbage</html>")

    provider = SarvamSTT(
        api_key="k",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(STTError, match="non-JSON"):
        provider.transcribe(PNG_BYTES, "audio/webm")


def test_post_multipart_requires_nonempty_audio():
    transport, calls = transport_with_status([200])
    with pytest.raises((ValueError, STTError)):
        post_multipart(
            url="http://test/x",
            headers={},
            file_field="file",
            filename="a",
            audio_bytes=b"",
            mime_type="audio/wav",
            form_fields={},
            timeout_seconds=1,
            transport=transport,
        )


# --- mock provider + /transcribe endpoint ----------------------------------

def test_mock_result_always_flagged():
    result = MockSTTProvider().transcribe(b"x", "audio/webm")
    assert result.is_mock is True
    assert result.provider == "mock"
    assert "[mock]" in result.text


@pytest.fixture
def mock_client() -> TestClient:
    app.dependency_overrides[stt_provider_dependency] = lambda: MockSTTProvider()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_transcribe_endpoint_works_in_mock_mode(mock_client):
    response = mock_client.post(
        "/transcribe",
        files={"file": ("clip.webm", PNG_BYTES, "audio/webm")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"]
    assert body["is_mock"] is True
    assert body["provider"] == "mock"
    assert "mock" in body["mock_note"].lower()
    assert body["stt_ms"] is not None and body["stt_ms"] >= 0


def test_transcribe_endpoint_rejects_empty_upload(mock_client):
    response = mock_client.post(
        "/transcribe",
        files={"file": ("empty.webm", b"", "audio/webm")},
    )
    assert response.status_code == 400


def test_transcribe_endpoint_rejects_oversized_upload(mock_client):
    big = b"x" * (26 * 1024 * 1024)
    response = mock_client.post(
        "/transcribe",
        files={"file": ("big.webm", big, "audio/webm")},
    )
    assert response.status_code == 413


def test_transcribe_endpoint_maps_stt_failure_to_502():
    class FailingProvider(MockSTTProvider):
        def transcribe(self, audio_bytes, mime_type):
            raise STTError("provider down")

    app.dependency_overrides[stt_provider_dependency] = lambda: FailingProvider()
    try:
        client = TestClient(app)
        response = client.post(
            "/transcribe",
            files={"file": ("c.webm", PNG_BYTES, "audio/webm")},
        )
        assert response.status_code == 502
    finally:
        app.dependency_overrides.clear()
