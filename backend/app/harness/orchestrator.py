"""The real pipeline: audio-or-text in -> STT -> retrieval -> generation out.

Stage policy (deliberate, per BUILD_PROMPT phase 6):
- STT failure is FATAL: no transcript means no query means nothing to serve.
  Raised as PipelineError so the endpoint can return a clear structured error.
- Retrieval failure is FATAL for the same reason (an ungrounded answer is the
  exact failure mode this system exists to prevent).
- Generation failure DEGRADES: the retrieved chunks are still real work worth
  returning, so respond with an empty answer + a warning instead of a 500.
- Missing chunk metadata is a WARNING, never a crash.

Every stage runs under stage_timer; the retriever's internal sub-trace is
merged into the request-level trace so one trace tells the whole story.
"""

import logging

from pydantic import BaseModel

from app.benchmarking.latency import GENERATION, STT, LatencyTrace, stage_timer
from app.config import Settings, get_settings
from app.generation.llm_client import (
    GenerationResult,
    LLMError,
    get_llm_provider,
)
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import RetrievedChunk
from app.stt.base import STTError, TranscriptResult
from app.stt.factory import get_stt_provider

logger = logging.getLogger(__name__)

# deterministic refusal used when retrieval found nothing — skipping the LLM
# here is both faster and more reliable than trusting the model to comply
NO_CONTEXT_REFUSAL = "I don't know based on the provided context."


class PipelineError(RuntimeError):
    """A non-recoverable pipeline stage failure; message is user-presentable."""

    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"{stage} failed: {detail}")
        self.stage = stage


class AskResponse(BaseModel):
    """Structured end-to-end response — no dict grab-bag."""

    transcript: str
    answer: str
    chunks: list[RetrievedChunk]
    latency_trace_ms: dict[str, float]
    retrieval_ms: float
    total_ms: float
    warnings: list[str] = []
    llm_provider: str
    llm_model: str
    llm_is_mock: bool
    stt_is_mock: bool = False


def _merge_trace(source: LatencyTrace, target: LatencyTrace) -> None:
    for stage, ms in source.stage_ms.items():
        target.record(stage, ms)


def _warn_on_missing_metadata(chunks: list[RetrievedChunk], warnings: list[str]) -> None:
    """One malformed hit must not sink a response with 4 good ones.

    metadata={} counts as incomplete — a chunk with no provenance can't be
    cited or audited later.
    """
    bad = [c.doc_id for c in chunks if not c.doc_id or not c.metadata]
    for doc_id in bad:
        warnings.append(f"retrieved chunk {doc_id!r} has incomplete metadata")


def run_pipeline(
    *,
    audio_bytes: bytes | None = None,
    mime_type: str = "audio/webm",
    text: str | None = None,
    settings: Settings | None = None,
    stt_provider=None,
    llm_provider=None,
) -> AskResponse:
    if (audio_bytes is None) == (text is None):
        raise ValueError("provide exactly one of audio_bytes or text")

    settings = settings or get_settings()
    trace = LatencyTrace()
    warnings: list[str] = []

    # --- stage 1: speech-to-text (skipped when the query arrives as text) ---
    if text is not None:
        transcript_result = TranscriptResult(text=text.strip(), provider="direct-input")
        stt_is_mock = False
    else:
        provider = stt_provider or get_stt_provider(settings)
        try:
            with stage_timer(STT, trace):
                transcript_result = provider.transcribe(audio_bytes or b"", mime_type)
        except STTError as exc:
            raise PipelineError("stt", str(exc)) from exc
        if not transcript_result.text.strip():
            raise PipelineError("stt", "empty transcript from provider")
        stt_is_mock = transcript_result.is_mock

    query = transcript_result.text

    # --- stage 2: retrieval (embed + vector search, internally timed) ---
    retriever = Retriever(settings=settings)
    try:
        retrieval = retriever.retrieve(query)
    except Exception as exc:
        raise PipelineError("retrieval", str(exc)[:200]) from exc
    _merge_trace(retrieval.trace, trace)

    chunks = retrieval.chunks
    _warn_on_missing_metadata(chunks, warnings)
    if not chunks:
        warnings.append("no context retrieved for this query")

    # --- stage 3: generation over the strict grounded prompt ---
    llm = llm_provider or get_llm_provider(settings)
    llm_is_mock = type(llm).__name__ == "MockLLM"
    llm_name = getattr(llm, "provider_name", None) or type(llm).__name__.replace(
        "LLM", ""
    ).lower()
    answer = ""
    generation: GenerationResult | None = None
    if not chunks:
        warnings.append("no context retrieved; skipped LLM call, returning deterministic refusal")
        with stage_timer(GENERATION, trace):
            answer = NO_CONTEXT_REFUSAL
    else:
        try:
            with stage_timer(GENERATION, trace):
                generation = llm.generate(query, chunks)
            answer = generation.answer
            llm_is_mock = generation.is_mock
        except LLMError as exc:
            # degrade, don't die: chunks are still valuable output
            msg = f"generation failed ({exc}); returning ungrounded-free empty answer"
            logger.warning(msg)
            warnings.append(msg)

    assert generation is None or isinstance(generation, GenerationResult)
    return AskResponse(
        transcript=query,
        answer=answer,
        chunks=chunks,
        latency_trace_ms=trace.stage_ms,
        retrieval_ms=trace.retrieval_ms(),
        total_ms=trace.total_ms(),
        warnings=warnings,
        llm_provider=llm_name,
        llm_model=getattr(llm, "model", getattr(llm, "_model_name", "unknown")),
        llm_is_mock=llm_is_mock,
        stt_is_mock=stt_is_mock,
    )
