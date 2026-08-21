"""LLM answer generation: Groq (default) / Gemini (fallback) / Mock.

One interface, config-selected:
    provider.generate(query, context_chunks) -> GenerationResult

Groq speaks the OpenAI SDK protocol, so the `openai` package works with just a
base_url override. Gemini uses google-genai, imported lazily so the package is
only required when LLM_PROVIDER=gemini is actually selected.

Retry/timeout policy mirrors the STT layer (stt/base.py): 3 attempts,
exponential backoff (0.5s -> 1s -> 2s), retrying timeouts / transport errors /
429 / 5xx; permanent failures (401 bad key, 400 bad request) fail fast. Every
failure surfaces as LLMError so the orchestrator can degrade gracefully.
"""

import logging
from typing import Callable, Protocol

import httpx
from openai import OpenAI
from pydantic import BaseModel
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings, get_settings
from app.generation.prompts import build_messages
from app.retrieval.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_MAX_ATTEMPTS = 3


class GenerationResult(BaseModel):
    """One completion outcome. is_mock=True MUST surface in any response."""

    answer: str
    provider: str
    model: str
    is_mock: bool = False


class LLMError(RuntimeError):
    """Raised after retries are exhausted or the failure is permanent."""


class _TransientStatus(Exception):
    """HTTP status worth retrying."""


RETRYABLE = (_TransientStatus, httpx.TimeoutException, httpx.TransportError)

# 429 rate-limit + server-side blips are worth another attempt; anything else
# (401 bad key, 400 bad request) is permanent
_TRANSIENT_CODES = {429, 500, 502, 503, 504}


def classify_sdk_error(exc: Exception) -> Exception:
    """Map an SDK failure to _TransientStatus (retryable) or LLMError (final).

    Works off attributes/message rather than importing each SDK's exception
    classes: the openai SDK and google-genai have different hierarchies that
    shift between versions, but both expose a numeric status/code on HTTP
    errors and mention timeouts/connection failures in their messages.
    """
    if isinstance(exc, (_TransientStatus, LLMError)):
        return exc
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and status in _TRANSIENT_CODES:
        return _TransientStatus(str(exc)[:200])
    msg = str(exc).lower()
    if "timed out" in msg or "timeout" in msg or "connection" in msg:
        return _TransientStatus(str(exc)[:200])
    return LLMError(f"LLM API call failed: {str(exc)[:200]}")


def _chat_with_retry(send_once: Callable[[], str]) -> str:
    """Run one chat-completion call under the shared retry/backoff policy.

    send_once() performs the network call and returns the answer text; it may
    raise RETRYABLE exceptions or LLMError for permanent ones. Transport-
    agnostic: the openai SDK and google-genai have different client shapes
    but both funnel failures through classify_sdk_error first.
    """
    try:
        for attempt in Retrying(
            retry=retry_if_exception_type(RETRYABLE),
            stop=stop_after_attempt(_MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
            reraise=True,
        ):
            with attempt:
                return send_once()
        raise AssertionError("unreachable: Retrying always returns or raises")
    except RETRYABLE as exc:
        raise LLMError(f"LLM API unreachable after {_MAX_ATTEMPTS} attempts: {exc}") from exc


class LLMProvider(Protocol):
    def generate(self, prompt: str, context_chunks: list[RetrievedChunk]) -> GenerationResult: ...


class GroqLLM:
    """OpenAI-SDK client pointed at Groq's compatible endpoint."""

    provider_name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        timeout_seconds: float = 30.0,
        http_transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GroqLLM requires an API key")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._transport = http_transport  # injectable for tests; None in prod
        self._client = OpenAI(
            api_key=api_key,
            base_url=GROQ_BASE_URL,
            timeout=timeout_seconds,
            max_retries=0,  # tenacity owns retries; never let the SDK double up
            http_client=httpx.Client(timeout=timeout_seconds, transport=http_transport)
            if http_transport is not None
            else None,
        )

    def generate(self, prompt: str, context_chunks: list[RetrievedChunk]) -> GenerationResult:
        messages = build_messages(prompt, context_chunks)

        def send_once() -> str:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,  # grounded QA: determinism beats creativity
                    max_tokens=512,
                )
            except Exception as exc:  # SDK boundary -> typed failure taxonomy
                raise classify_sdk_error(exc) from exc
            choice = response.choices[0] if response.choices else None
            content = (choice.message.content or "").strip() if choice else ""
            if not content:
                # empty at temperature=0 is deterministic (length cap/filter) —
                # retrying cannot change it, so fail permanently
                raise LLMError("model returned an empty completion")
            return content

        answer = _chat_with_retry(send_once=send_once)
        return GenerationResult(answer=answer, provider="groq", model=self.model)


class GeminiLLM:
    """google-genai client; import deferred so Groq-only installs don't need it."""

    provider_name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("GeminiLLM requires an API key")
        try:
            from google import genai  # lazy: only needed when actually selected
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise LLMError(
                "LLM_PROVIDER=gemini requires the 'google-genai' package "
                "(pip install google-genai)"
            ) from exc
        self._model_name = model
        self._client = genai.Client(api_key=api_key, http_options={"timeout": timeout_seconds})

    def generate(self, prompt: str, context_chunks: list[RetrievedChunk]) -> GenerationResult:
        messages = build_messages(prompt, context_chunks)
        # keep the strict grounding rules as a proper system instruction
        # instead of flattening it into the user turn
        config = {
            "temperature": 0.0,
            "max_output_tokens": 512,
            "system_instruction": messages[0]["content"],
        }
        user_turn = messages[1]["content"]

        def send_once() -> str:
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=user_turn,
                    config=config,
                )
            except Exception as exc:  # SDK boundary -> typed failure taxonomy
                raise classify_sdk_error(exc) from exc
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise LLMError("model returned an empty completion")
            return text

        answer = _chat_with_retry(send_once=send_once)
        return GenerationResult(answer=answer, provider="gemini", model=self._model_name)


class MockLLM:
    """Deterministic offline generation for dev/tests — extracts from context.

    Never pretends to be real: is_mock=True always, loud warning at
    construction, answer prefixed so it can't pass for model output.
    """

    provider_name = "mock"

    def __init__(self) -> None:
        logger.warning(
            "LLM running in MOCK mode — answers are extracted from retrieved "
            "chunks, not generated by a model. Do not demo this as real."
        )
        self.model = "mock-extractive"

    def generate(self, prompt: str, context_chunks: list[RetrievedChunk]) -> GenerationResult:
        top = next((c.text.strip() for c in context_chunks if c.text.strip()), "")
        if not top:
            answer = "[mock] I don't know based on the provided context."
        else:
            answer = f"[mock] Based on the retrieved context: {top[:300]}"
        return GenerationResult(
            answer=answer, provider="mock", model=self.model, is_mock=True
        )


def get_llm_provider(settings: Settings | None = None):
    """Same policy as the STT factory: missing key degrades to MOCK loudly;
    unknown provider NAME is a hard error, not something to mask."""
    settings = settings or get_settings()
    name = settings.llm_provider.strip().lower()

    if name == "mock":
        return MockLLM()
    if name == "groq":
        if settings.groq_api_key:
            return GroqLLM(api_key=settings.groq_api_key, model=settings.groq_model)
        logger.warning("LLM_PROVIDER=groq but GROQ_API_KEY is empty — using MOCK LLM")
        return MockLLM()
    if name == "gemini":
        if settings.gemini_api_key:
            return GeminiLLM(api_key=settings.gemini_api_key)
        logger.warning("LLM_PROVIDER=gemini but GEMINI_API_KEY is empty — using MOCK LLM")
        return MockLLM()

    raise ValueError(
        f"unknown LLM_PROVIDER {settings.llm_provider!r} (expected groq/gemini/mock)"
    )
