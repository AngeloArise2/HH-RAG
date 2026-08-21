from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.benchmarking.latency import LatencyTrace, STT as STT_STAGE, stage_timer
from app.stt.base import STTError, STTProvider
from app.stt.factory import get_stt_provider

app = FastAPI(title="voice-rag", version="0.1.0")

# providers' sync STT endpoints accept ~30s clips; reject anything absurd
# before it burns a network call (Sarvam 400s on oversized bodies anyway)
MAX_AUDIO_BYTES = 25 * 1024 * 1024


def stt_provider_dependency() -> STTProvider:
    """Resolved per request so config changes / test overrides take effect."""
    return get_stt_provider()


class TranscribeResponse(BaseModel):
    text: str
    provider: str
    is_mock: bool
    language: str | None = None
    stt_ms: float | None = None
    mock_note: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    provider: STTProvider = Depends(stt_provider_dependency),
) -> TranscribeResponse:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio upload")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="audio upload too large")

    mime_type = file.content_type or "audio/webm"
    trace = LatencyTrace()
    try:
        # blocking network I/O — keep it off the event loop; the stage_timer
        # still records if the call raises after exhausting retries
        with stage_timer(STT_STAGE, trace):
            result = await run_in_threadpool(
                provider.transcribe, audio_bytes, mime_type
            )
    except STTError as exc:
        raise HTTPException(status_code=502, detail=f"STT failed: {exc}") from exc

    mock_note = (
        "MOCK TRANSCRIPT — no real speech recognition ran "
        "(provider key missing or STT_PROVIDER=mock)"
        if result.is_mock
        else None
    )
    return TranscribeResponse(
        text=result.text,
        provider=result.provider,
        is_mock=result.is_mock,
        language=result.language,
        stt_ms=trace.stage_ms.get(STT_STAGE),
        mock_note=mock_note,
    )
