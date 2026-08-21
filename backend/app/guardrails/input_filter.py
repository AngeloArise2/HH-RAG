"""Input-side guardrail: off-topic + unsafe detection BEFORE retrieval.

One cheap strict-schema LLM call (temp 0, ~16 output tokens) classifies the
transcribed query. Runs inside the orchestrator's request path; tripping it
short-circuits the pipeline before any embedding or vector search happens.

Scope definition is deliberately GENEROUS: anything plausibly answerable from
a general web-passage corpus (MSMARCO-style reference text) is ON_TOPIC.
A guardrail that blocks normal questions is worse than no guardrail.

Failure policy: if the guard call itself fails after retries (LLMError), we
fail OPEN with a loud warning — a transient guard outage must not brick the
whole pipeline, and the failure is visible in warnings + logs either way.
"""

import logging
import re

from pydantic import BaseModel

from app.generation.llm_client import LLMError
from app.guardrails.refusal import refusal_for

logger = logging.getLogger(__name__)


class GuardVerdict(BaseModel):
    verdict: str  # on_topic | off_topic | unsafe (filter) / supported | partial | unsupported (judge)
    reason: str = ""
    failed_open: bool = False  # True = the guard itself failed and we passed anyway


CLASSIFY_SYSTEM = """\
You classify user questions for a question-answering system grounded in a \
corpus of general web passages (reference-style text about business, \
history, science, everyday life).

Classify the question as exactly one of:
- ON_TOPIC: plausibly answerable from static reference passages. When in \
doubt, choose this.
- OFF_TOPIC: not answerable from static reference passages — personal or \
real-time info ("what's the time", "who am I"), chit-chat, gibberish, or \
meta-questions about the system itself.
- UNSAFE: requests for serious harm (weapons/explosives/violence how-tos), \
hate or harassment, or sexual content.

Respond with ONLY one word: ON_TOPIC, OFF_TOPIC, or UNSAFE."""

# tolerant of "ON_TOPIC", "off-topic", "Off Topic" spellings
_VERDICT_PATTERN = re.compile(r"\b(on[\s_-]?topic|off[\s_-]?topic|unsafe)\b")


def _parse_verdict(text: str) -> str:
    """Extract the verdict token from a possibly chatty response."""
    match = _VERDICT_PATTERN.search(text.lower())
    return re.sub(r"[\s\-]+", "_", match.group(0)) if match else ""


def classify_input(query: str, llm) -> GuardVerdict:
    try:
        raw = llm.complete_raw(CLASSIFY_SYSTEM, query.strip(), max_tokens=128)
    except LLMError as exc:
        logger.warning("input_filter guard call failed, failing open: %s", exc)
        return GuardVerdict(
            verdict="on_topic", reason=f"guard unavailable: {exc}", failed_open=True
        )

    verdict = _parse_verdict(raw)
    if verdict not in {"on_topic", "off_topic", "unsafe"}:
        logger.warning("input_filter got unparseable response %r, failing open", raw[:80])
        return GuardVerdict(
            verdict="on_topic", reason="guard response unparseable", failed_open=True
        )
    return GuardVerdict(verdict=verdict)


def is_refusal(verdict: GuardVerdict) -> bool:
    return verdict.verdict in {"off_topic", "unsafe"}


def refusal_reason(verdict: GuardVerdict) -> str:
    return refusal_for(verdict.verdict)
