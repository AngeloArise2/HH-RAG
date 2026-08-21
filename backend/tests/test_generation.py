"""LLM client tests: strict prompt, retry taxonomy, factory policy, mock.

Network behavior runs through an injected httpx.MockTransport inside the
openai SDK client (GroqLLM accepts http_transport) — zero real HTTP here.
"""

import logging

import httpx
import pytest

from app.config import Settings
from app.generation.llm_client import (
    GenerationResult,
    GroqLLM,
    LLMError,
    MockLLM,
    classify_sdk_error,
    get_llm_provider,
)
from app.generation.prompts import build_context_block, build_messages
from app.retrieval.vector_store import RetrievedChunk

# --- helpers ----------------------------------------------------------------


def make_chunks(n=2) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"c{i}",
            doc_id=f"d{i}",
            text=f"context passage number {i} about incorporation rules",
            score=0.9 - i * 0.1,
            metadata={"doc_id": f"d{i}"},
        )
        for i in range(n)
    ]


def groq_with_transport(handler) -> tuple["GroqLLM", dict]:
    """Build GroqLLM around a MockTransport; return provider + call counter."""
    calls = {"count": 0}

    def counting_handler(request):
        calls["count"] += 1
        return handler(request)

    provider = GroqLLM(
        api_key="k-test",
        http_transport=httpx.MockTransport(counting_handler),
    )
    return provider, calls


def ok_completion(text="grounded answer") -> dict:
    return {
        "id": "x",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
    }


# --- prompts ----------------------------------------------------------------


def test_prompt_is_genuinely_strict():
    messages = build_messages("what is the fee?", make_chunks())
    system = messages[0]["content"]
    assert "ONLY" in system
    assert "Do NOT" in system or "do not" in system.lower()
    assert "I don't know" in system
    user = messages[1]["content"]
    # every chunk text must actually be in the context block
    assert "context passage number 0" in user
    assert "context passage number 1" in user
    assert "[1]" in user and "[2]" in user
    assert "what is the fee?" in user


def test_empty_chunk_list_still_builds_valid_prompt():
    user = build_messages("q", [])[1]["content"]
    assert "no context retrieved" in user


def test_whitespace_chunks_are_dropped_from_context():
    chunk = RetrievedChunk(chunk_id="c", doc_id="d", text="   ", score=0.5)
    assert "(no context retrieved)" in build_context_block([chunk])


# --- failure classification (shared by both SDKs) ---------------------------


class _FakeStatus(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


def test_transient_status_codes_classified_retryable():
    for code in (429, 500, 502, 503, 504):
        mapped = classify_sdk_error(_FakeStatus(code))
        assert type(mapped).__name__ == "_TransientStatus"


def test_permanent_status_and_unknown_errors_are_final():
    assert type(classify_sdk_error(_FakeStatus(401))).__name__ == "LLMError"
    assert type(classify_sdk_error(ValueError("bad"))).__name__ == "LLMError"


def test_timeout_and_connection_messages_are_retryable():
    assert type(classify_sdk_error(RuntimeError("Request timed out"))).__name__ == "_TransientStatus"
    assert type(classify_sdk_error(RuntimeError("Connection error."))).__name__ == "_TransientStatus"


def test_existing_typed_errors_pass_through_unchanged():
    err = LLMError("already typed")
    assert classify_sdk_error(err) is err


# --- GroqLLM via mock transport ---------------------------------------------


def test_groq_generate_success_parses_answer():
    provider, calls = groq_with_transport(lambda req: httpx.Response(200, json=ok_completion()))
    result = provider.generate("query", make_chunks())
    assert calls["count"] == 1
    assert isinstance(result, GenerationResult)
    assert result.answer == "grounded answer"
    assert result.provider == "groq"
    assert result.is_mock is False


def test_groq_retries_transient_then_succeeds():
    state = {"n": 0}

    def handler(req):
        state["n"] += 1
        if state["n"] <= 2:
            return httpx.Response(503, json={"error": "overloaded"})
        return httpx.Response(200, json=ok_completion())

    provider, calls = groq_with_transport(handler)
    result = provider.generate("q", make_chunks())
    assert calls["count"] == 3
    assert result.answer == "grounded answer"


def test_groq_exhausts_retries_into_llm_error():
    provider, calls = groq_with_transport(
        lambda req: httpx.Response(500, json={"error": "boom"})
    )
    with pytest.raises(LLMError, match="unreachable after 3 attempts"):
        provider.generate("q", make_chunks())
    assert calls["count"] == 3


def test_groq_bad_key_fails_fast_without_retry():
    provider, calls = groq_with_transport(
        lambda req: httpx.Response(401, json={"error": "invalid key"})
    )
    with pytest.raises(LLMError, match="401"):
        provider.generate("q", make_chunks())
    assert calls["count"] == 1


def test_groq_empty_completion_is_permanent_not_retried():
    provider, calls = groq_with_transport(
        lambda req: httpx.Response(200, json={"id": "x", "choices": []})
    )
    with pytest.raises(LLMError, match="empty completion"):
        provider.generate("q", make_chunks())
    assert calls["count"] == 1


# --- factory + mock ---------------------------------------------------------


def test_factory_returns_mock_when_key_missing(caplog):
    with caplog.at_level(logging.WARNING):
        llm = get_llm_provider(Settings(llm_provider="groq", groq_api_key=""))
    assert isinstance(llm, MockLLM)
    assert any("MOCK" in r.message for r in caplog.records)


def test_factory_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        get_llm_provider(Settings(llm_provider="chatgpt"))


def test_mock_llm_flags_itself_and_uses_context():
    result = MockLLM().generate("q", make_chunks(1))
    assert result.is_mock is True
    assert "[mock]" in result.answer
    assert "context passage number 0" in result.answer


def test_mock_llm_without_context_says_idontknow():
    result = MockLLM().generate("q", [])
    assert "don't know" in result.answer
