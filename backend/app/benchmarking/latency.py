"""Centralized latency measurement for the whole pipeline.

Every timed operation goes through `stage_timer` recording into a
`LatencyTrace` — no ad-hoc time.time()/perf_counter calls scattered around
the codebase. The benchmark script (phase 8) and any future profiling read
these traces, so the mechanism must stay uniform.

Canonical stage names (add new ones here AND wrap them in stage_timer):
    stt, embed_query, vector_search, chunk_assembly, generation, guardrail_check

Retrieval-only latency (what gets checked against the 200ms target) =
embed_query + vector_search + chunk_assembly. Everything else belongs to the
full end-to-end number, which is reported honestly without a target.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

from pydantic import BaseModel, Field

STT = "stt"
EMBED_QUERY = "embed_query"
VECTOR_SEARCH = "vector_search"
CHUNK_ASSEMBLY = "chunk_assembly"
GENERATION = "generation"
GUARDRAIL_CHECK = "guardrail_check"

CANONICAL_STAGES = (
    STT,
    EMBED_QUERY,
    VECTOR_SEARCH,
    CHUNK_ASSEMBLY,
    GENERATION,
    GUARDRAIL_CHECK,
)
RETRIEVAL_STAGES = (EMBED_QUERY, VECTOR_SEARCH, CHUNK_ASSEMBLY)


class LatencyTrace(BaseModel):
    """Durations (ms) per stage for ONE request. Repeated stages accumulate."""

    stage_ms: dict[str, float] = Field(default_factory=dict)

    def record(self, stage: str, duration_ms: float) -> None:
        self.stage_ms[stage] = round(
            self.stage_ms.get(stage, 0.0) + duration_ms, 3
        )

    def retrieval_ms(self) -> float:
        """Sum of the retrieval-side stages — this is checked vs 200ms."""
        return round(sum(v for k, v in self.stage_ms.items() if k in RETRIEVAL_STAGES), 3)

    def total_ms(self) -> float:
        return round(sum(self.stage_ms.values()), 3)

    def as_table(self) -> list[tuple[str, float]]:
        return [(k, v) for k, v in self.stage_ms.items()]


_current_trace: ContextVar[Optional[LatencyTrace]] = ContextVar(
    "current_latency_trace", default=None
)


def set_current_trace(trace: LatencyTrace) -> Any:
    """Attach a trace to the current request context; returns a reset token."""
    return _current_trace.set(trace)


def reset_current_trace(token: Any) -> None:
    _current_trace.reset(token)


def current_trace() -> Optional[LatencyTrace]:
    return _current_trace.get()


@contextmanager
def stage_timer(stage: str, trace: LatencyTrace | None = None) -> Iterator[Optional[LatencyTrace]]:
    """Time one pipeline stage in milliseconds.

    Records into `trace`, or into the current request-context trace when
    omitted. Recording happens even if the body raises — a failed LLM call is
    still latency worth seeing.
    """
    target = trace if trace is not None else _current_trace.get()
    start = time.perf_counter()
    try:
        yield target
    finally:
        if target is not None:
            target.record(stage, (time.perf_counter() - start) * 1000.0)
