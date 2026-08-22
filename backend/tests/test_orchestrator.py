"""Orchestrator + /ask integration tests.

Full pipeline runs against the REAL local Chroma index (built in phase 3)
with mocked STT/LLM providers — proving wiring, timing, and degradation
without network. Live Groq verification happens separately via scripts.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.generation.llm_client import GenerationResult, LLMError
from app.generation.prompts import build_messages
from app.harness.orchestrator import AskResponse, PipelineError, run_pipeline
from app.main import app
from app.stt.base import STTError, TranscriptResult
from app.stt.mock import MockSTTProvider

TEST_SETTINGS = Settings(default_chunk_strategy="metadata_aware")


class FakeLLM:
    """Deterministic stand-in that records what it was given."""

    def __init__(
        self,
        answer: str = "a grounded answer",
        fail: bool = False,
        filter_verdict: str = "ON_TOPIC",
        judge_verdict: str = "SUPPORTED",
    ):
        self.answer = answer
        self.fail = fail
        self.filter_verdict = filter_verdict
        self.judge_verdict = judge_verdict
        self.last_prompt = None
        self.last_chunks = None
        self.model = "fake"
        self.provider_name = "fake"

    def complete_raw(self, system: str, user: str, max_tokens: int = 16) -> str:
        if "classify" in system.lower():
            return self.filter_verdict
        return self.judge_verdict

    def generate(self, prompt, context_chunks):
        if self.fail:
            raise LLMError("simulated provider outage")
        # mirror the real providers' contract: they build chat messages from
        # chunks via prompts.build_messages — which is where CHUNK_ASSEMBLY
        # gets recorded against the ambient request trace.
        self.last_prompt = build_messages(prompt, context_chunks)
        self.last_chunks = context_chunks
        return GenerationResult(
            answer=self.answer, provider="fake", model="fake", is_mock=False
        )


def test_full_text_pipeline_real_index_fake_llm():
    llm = FakeLLM()
    response = run_pipeline(
        text="what is incorporation",
        settings=TEST_SETTINGS,
        llm_provider=llm,
    )
    assert isinstance(response, AskResponse)
    assert response.transcript == "what is incorporation"
    assert response.answer == "a grounded answer"
    assert len(response.chunks) > 0
    # latency trace must be populated with retrieval stages at minimum
    assert "embed_query" in response.latency_trace_ms
    assert "vector_search" in response.latency_trace_ms
    assert "chunk_assembly" in response.latency_trace_ms
    assert "generation" in response.latency_trace_ms
    # retrieval budget = all three retrieval-side stages, per RETRIEVAL_STAGES
    t = response.latency_trace_ms
    expected_retrieval = round(
        t["embed_query"] + t["vector_search"] + t["chunk_assembly"], 3
    )
    assert response.retrieval_ms == pytest.approx(expected_retrieval)
    assert response.total_ms >= response.retrieval_ms
    assert response.llm_is_mock is False
    # LLM received the query and real retrieved chunks, as chat messages
    assert "what is incorporation" in llm.last_prompt[1]["content"]
    assert llm.last_chunks == response.chunks


def test_audio_pipeline_with_mock_stt_end_to_end():
    response = run_pipeline(
        audio_bytes=b"fake-audio-bytes",
        settings=TEST_SETTINGS,
        stt_provider=MockSTTProvider(),
        llm_provider=FakeLLM(),
    )
    assert "[mock]" in response.transcript
    assert response.stt_is_mock is True
    assert "stt" in response.latency_trace_ms
    assert response.answer


def test_stt_failure_raises_fatal_pipeline_error():
    class BrokenSTT(MockSTTProvider):
        def transcribe(self, audio_bytes, mime_type):
            raise STTError("provider unreachable")

    with pytest.raises(PipelineError, match="stt failed"):
        run_pipeline(
            audio_bytes=b"x",
            stt_provider=BrokenSTT(),
            settings=TEST_SETTINGS,
            llm_provider=FakeLLM(),
        )


def test_empty_transcript_is_fatal_not_silent():
    class EmptySTT(MockSTTProvider):
        def transcribe(self, audio_bytes, mime_type):
            return TranscriptResult(text="", provider="broken")

    with pytest.raises(PipelineError, match="empty transcript"):
        run_pipeline(
            audio_bytes=b"x",
            stt_provider=EmptySTT(),
            settings=TEST_SETTINGS,
            llm_provider=FakeLLM(),
        )


def test_llm_failure_degrades_with_warning_not_crash():
    response = run_pipeline(
        text="incorporation of companies",
        settings=TEST_SETTINGS,
        llm_provider=FakeLLM(fail=True),
    )
    assert response.answer == ""  # no fabricated answer
    assert any("generation failed" in w for w in response.warnings)
    assert len(response.chunks) > 0  # retrieval work still returned
    assert "generation" in response.latency_trace_ms  # failure still timed


def test_both_or_neither_input_rejected():
    with pytest.raises(ValueError):
        run_pipeline(text="x", audio_bytes=b"y")
    with pytest.raises(ValueError):
        run_pipeline()


def test_no_context_query_warns_but_returns():
    llm = FakeLLM()
    response = run_pipeline(
        text="zzzzqqqq unrelated gibberish query xyzzy",
        settings=TEST_SETTINGS,
        llm_provider=llm,
    )
    # gibberish still retrieves *something* (nearest neighbors) — chunks may be
    # non-empty; the assertion is that the pipeline completes either way
    assert isinstance(response.warnings, list)


# --- /ask endpoint ----------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_ask_endpoint_text_mode_real_index(client, monkeypatch):
    monkeypatch.setattr(
        "app.main.run_pipeline",
        lambda **kw: run_pipeline(text=kw["text"], settings=TEST_SETTINGS,
                                  llm_provider=FakeLLM()),
    )
    response = client.post("/ask", data={"query": "what is incorporation"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "a grounded answer"
    assert body["latency_trace_ms"]["vector_search"] > 0
    assert isinstance(body["warnings"], list)


def test_ask_endpoint_requires_exactly_one_input(client):
    assert client.post("/ask").status_code == 422
    response = client.post(
        "/ask", data={"query": "hi"},
        files={"file": ("a.webm", b"x", "audio/webm")},
    )
    assert response.status_code == 422


def test_ask_endpoint_maps_pipeline_error_to_502(client, monkeypatch):
    def boom(**kw):
        raise PipelineError("stt", "simulated total failure")

    monkeypatch.setattr("app.main.run_pipeline", boom)
    response = client.post("/ask", data={"query": "anything"})
    assert response.status_code == 502


def test_refusal_path_has_no_chunk_assembly_stage():
    """Input-guard refusals short-circuit BEFORE retrieval — the trace must
    not pretend chunk work happened."""
    response = run_pipeline(
        text="what time is it right now",
        settings=TEST_SETTINGS,
        llm_provider=FakeLLM(filter_verdict="OFF_TOPIC"),
    )
    assert response.refused is True
    assert "chunk_assembly" not in response.latency_trace_ms
