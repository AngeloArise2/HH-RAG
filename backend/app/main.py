import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.benchmarking.latency import LatencyTrace, STT as STT_STAGE, stage_timer
from app.config import get_settings
from app.harness.orchestrator import AskResponse, PipelineError, run_pipeline
from app.retrieval.retriever import warm_retrieval
from app.stt.base import STTError, STTProvider
from app.stt.factory import get_stt_provider

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the retrieval path at startup: the first embed otherwise costs
    ~9-11s (MiniLM load), which must never land on the first real query."""
    try:
        warm_ms = await run_in_threadpool(warm_retrieval)
        logger.info("retrieval warmed up in %.0fms — ready", warm_ms)
    except Exception as exc:  # warmup must never block startup
        logger.warning("startup retrieval warmup skipped: %s", exc)
    yield


app = FastAPI(title="voice-rag", version="0.1.0", lifespan=lifespan)

# Dev convenience only: when the frontend is served by THIS service (prod),
# requests are same-origin and CORS never triggers. The Vite dev server runs
# on :5173, so that origin (plus anything configured) gets allowed here.
_origins = [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.post("/ask", response_model=AskResponse)
async def ask(
    file: UploadFile | None = File(None),
    query: str | None = Form(None),
) -> AskResponse:
    """End-to-end pipeline: audio (transcribed) or raw text -> grounded answer.

    Accepts multipart form with EITHER a `file` (audio) OR a `query` text
    field — text exists so the pipeline is demoable/testable without a mic.
    """
    if (file is None) == (query is None):
        raise HTTPException(
            status_code=422,
            detail="provide exactly one of: audio file, or query text",
        )

    try:
        if file is not None:
            audio_bytes = await file.read()
            if not audio_bytes:
                raise HTTPException(status_code=400, detail="empty audio upload")
            if len(audio_bytes) > MAX_AUDIO_BYTES:
                raise HTTPException(status_code=413, detail="audio upload too large")
            return await run_in_threadpool(
                run_pipeline,
                audio_bytes=audio_bytes,
                mime_type=file.content_type or "audio/webm",
            )
        return await run_in_threadpool(run_pipeline, text=query)
    except PipelineError as exc:
        # stage can't proceed — structured 502, not a stack trace
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# --- frontend serving (single-service deploy) --------------------------------
# Mounted LAST so /ask, /transcribe and /health always match first. Only
# active when a built frontend exists (docker image or `npm run build`
# locally); absent dir = pure-API mode for tests/dev.
_DIST_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST_DIR), html=True), name="frontend")
