"""Request-path retriever: embed + search, each stage timed separately.

Embedding the query (local model forward pass) and searching the vector
index (ANN traversal) have completely different cost profiles — merging them
into one blob would hide exactly the number we need to defend the 200ms
target. Hence two stage_timer blocks, always.
"""

from pydantic import BaseModel

from app.benchmarking.latency import EMBED_QUERY, VECTOR_SEARCH, LatencyTrace, stage_timer
from app.config import Settings, get_settings
from app.retrieval import embed as embed_module
from app.retrieval import vector_store
from app.retrieval.vector_store import RetrievedChunk


class RetrievalResult(BaseModel):
    """Structured retrieval response: hits plus their per-stage latency trace."""

    query: str
    strategy: str
    chunks: list[RetrievedChunk]
    trace: LatencyTrace


class Retriever:
    """Searches one strategy's collection; strategy is swappable via config."""

    def __init__(
        self,
        strategy_name: str | None = None,
        settings: Settings | None = None,
        top_k: int = 5,
    ) -> None:
        self.settings = settings or get_settings()
        self.strategy_name = strategy_name or self.settings.default_chunk_strategy
        self.top_k = top_k

    def retrieve(self, query_text: str) -> RetrievalResult:
        trace = LatencyTrace()
        with stage_timer(EMBED_QUERY, trace):
            vectors = embed_module.embed_texts([query_text])
        with stage_timer(VECTOR_SEARCH, trace):
            hits = vector_store.search_vectors(
                vectors[0],
                self.strategy_name,
                top_k=self.top_k,
                settings=self.settings,
            )
        return RetrievalResult(
            query=query_text, strategy=self.strategy_name, chunks=hits, trace=trace
        )
