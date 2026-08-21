"""The real pipeline: audio-or-text in -> STT -> guard -> retrieval ->
generation -> grounding guard -> out.

Stage policy (deliberate, per BUILD_PROMPT phases 6-7):
- STT failure is FATAL: no transcript means no query means nothing to serve.
  Raised as PipelineError so the endpoint can return a clear structured error.
- INPUT GUARD (off-topic/unsafe) runs BEFORE retrieval and short-circuits the
  request on a trip — no embedding, no vector search, no generation.
- Retrieval failure is FATAL for grounding reasons (no context = no honest
  answer); empty-but-successful retrieval short-circuits to a deterministic
  don't-know refusal without burning an LLM call.
- Generation failure DEGRADES: the retrieved chunks are still real work worth
  returning, so respond with an empty answer + a warning instead of a 500.
- GROUNDING GUARD runs AFTER generation and BEFORE response assembly; an
  UNSUPPORTED verdict replaces the answer with a refusal. The fabricated
  text is never returned to the caller — it stays in warnings only if at all.
- Missing chunk metadata is a WARNING, never a crash.

Every stage runs under stage_timer; the retriever's internal sub-trace is
merged into the request-level trace so one trace tells the whole story.
"""

import logging

from pydantic import BaseModel

from app.benchmarking.latency import (
    GENERATION,
    GUARDRAIL_CHECK,
    STT,
    LatencyTrace,
    stage_timer,
)
from app.config import Settings, get_settings
from app.generation.llm_client import (
    GenerationResult,
    LLMError,
    get_llm_provider,
)
from app.guardrails.grounding_check import check_grounded, is_unsupported
from app.guardrails.input_filter import GuardVerdict, classify_input, is_refusal
from app.guardrails.refusal import refusal_for
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
    """Structured end-to-end response — no dict grab-bag.

    grounding_verified and refused are INDEPENDENT fields covering different
    failure modes: refused=True means a guard deliberately blocked the answer;
    grounding_verified=False means an answer IS returned but the judge could
    not verify it (guard outage). Only meaningful when refused=False.
    """

    transcript: str
    answer: str
    refused: bool = False
    refusal_reason: str | None = None
    grounding_verified: bool = True
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

    # resolve the LLM provider once; guards + generation share it
    llm = llm_provider or get_llm_provider(settings)
    llm_name = getattr(llm, "provider_name", None) or type(llm).__name__.replace(
        "LLM", ""
    ).lower()
    llm_model = getattr(llm, "model", getattr(llm, "_model_name", "unknown"))

    # --- stage 2: INPUT GUARD — off-topic/unsafe, before any retrieval work ---
    with stage_timer(GUARDRAIL_CHECK, trace):
        input_verdict: GuardVerdict = classify_input(query, llm)
    if input_verdict.failed_open:
        warnings.append(f"input filter failed open: {input_verdict.reason}")
    if is_refusal(input_verdict):
        logger.info("input guard tripped (%s): %s", input_verdict.verdict, query[:80])
        return AskResponse(
            transcript=query,
            answer=refusal_for(input_verdict.verdict),
            refused=True,
            refusal_reason=input_verdict.verdict,
            chunks=[],
            latency_trace_ms=trace.stage_ms,
            retrieval_ms=trace.retrieval_ms(),
            total_ms=trace.total_ms(),
            warnings=[f"input filter refused the query ({input_verdict.verdict})"],
            llm_provider=llm_name,
            llm_model=llm_model,
            llm_is_mock=type(llm).__name__ == "MockLLM",
            stt_is_mock=stt_is_mock,
        )

    # --- stage 3: retrieval (embed + vector search, internally timed) ---
    retriever = Retriever(settings=settings)
    try:
        retrieval = retriever.retrieve(query)
    except Exception as exc:
        raise PipelineError("retrieval", str(exc)[:200]) from exc
    _merge_trace(retrieval.trace, trace)

    chunks = retrieval.chunks
    _warn_on_missing_metadata(chunks, warnings)

    # --- stage 4: generation over the strict grounded prompt ---
    llm_is_mock = type(llm).__name__ == "MockLLM"
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

    # --- stage 5: GROUNDING GUARD — after generation, before assembly ---
    refused = False
    refusal_reason_str = None
    grounding_verified = True  # only flips when a judge ran and could not verify
    if generation is not None and answer:
        with stage_timer(GUARDRAIL_CHECK, trace):
            judge = check_grounded(answer, chunks, llm)
        if not judge.verified:
            grounding_verified = False
            warnings.append(f"grounding check failed open: {judge.reason}")
        if judge.failed_open:
            logger.warning(
                "grounding judge unavailable for %r — answer returned unverified",
                query[:80],
            )
        if is_unsupported(judge):
            logger.warning("grounding judge rejected answer for %r", query[:80])
            warnings.append(
                f"grounding check rejected the generated answer ({judge.reason or 'unsupported'})"
            )
            answer = refusal_for("ungrounded")
            refused = True
            refusal_reason_str = "ungrounded"

    return AskResponse(
        transcript=query,
        answer=answer,
        refused=refused,
        refusal_reason=refusal_reason_str,
        grounding_verified=grounding_verified,
        chunks=chunks,
        latency_trace_ms=trace.stage_ms,
        retrieval_ms=trace.retrieval_ms(),
        total_ms=trace.total_ms(),
        warnings=warnings,
        llm_provider=llm_name,
        llm_model=llm_model,
        llm_is_mock=llm_is_mock,
        stt_is_mock=stt_is_mock,
    )
