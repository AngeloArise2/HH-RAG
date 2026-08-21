"""Output-side guardrail: is the generated answer actually IN the context?

LLM-as-judge with a strict one-word verdict, run AFTER generation and BEFORE
the response is assembled. This is the second, independent line of defense —
the prompt template (prompts.py) is the first. A model that ignores its
grounding instructions still gets caught here.

Only UNSUPPORTED trips a refusal; PARTIAL passes through (over-refusing
half-supported answers would make the system unusable).

Failure policy matches input_filter: guard call fails -> fail OPEN with a
loud warning rather than brick the pipeline on a guard outage.
"""

import logging
import re

from pydantic import BaseModel

from app.generation.llm_client import LLMError
from app.generation.prompts import build_context_block
from app.guardrails.input_filter import (  # noqa: F401 (GuardVerdict is a shared type)
    GuardVerdict,
    match_is_negated,
)
from app.guardrails.refusal import UNGROUNDED_REFUSAL

logger = logging.getLogger(__name__)


class JudgeVerdict(BaseModel):
    verdict: str  # supported | partial | unsupported | "" (no verdict obtained)
    reason: str = ""
    failed_open: bool = False
    # False ONLY when the judge itself could not produce a verdict (API
    # failure, unparseable response). The orchestrator surfaces this on
    # AskResponse.grounding_verified so clients can distinguish "judged and
    # passed" from "never judged" without string-matching warnings.
    verified: bool = True


JUDGE_SYSTEM = """\
You are a strict grounding judge. You will see numbered context passages \
and a proposed answer.

Decide whether the answer is factually supported by the passages ONLY:
- SUPPORTED: every factual claim in the answer appears in, or follows \
directly from, the context passages.
- PARTIAL: some claims are grounded but others are not.
- UNSUPPORTED: the answer contradicts the passages, or its key claims rely \
on knowledge that is not in them.

Respond with ONLY one word: SUPPORTED, PARTIAL, or UNSUPPORTED."""

# \b word boundaries + longest-token-first alternation: "unsupported" must
# win over its substring "supported", never the other way round
_JUDGE_PATTERN = re.compile(r"\b(unsupported|partial|supported)\b")


def parse_judge_verdict(text: str) -> str:
    lowered = text.lower().strip()
    # exact single-token responses need no interpretation
    if lowered in {"supported", "partial", "unsupported"}:
        return lowered
    match = _JUDGE_PATTERN.search(lowered)
    if not match or match_is_negated(lowered, match.start()):
        # unparseable / negated ("NOT SUPPORTED") -> "" -> caller fails open
        return ""
    return match.group(0)


def _judge_user_text(answer: str, chunks) -> str:
    return (
        f"Context passages:\n{build_context_block(chunks)}\n\n"
        f"Proposed answer:\n{answer}"
    )


def check_grounded(answer: str, chunks, llm) -> JudgeVerdict:
    try:
        raw = llm.complete_raw(
            JUDGE_SYSTEM,
            _judge_user_text(answer, chunks),
            max_tokens=128,
        )
    except LLMError as exc:
        logger.warning("grounding_check judge call failed, failing open: %s", exc)
        # structured fail-open: no verdict obtained, answer passes, and the
        # orchestrator flags grounding_verified=False on the response
        return JudgeVerdict(
            verdict="",
            reason=str(exc)[:200],
            failed_open=True,
            verified=False,
        )

    verdict = parse_judge_verdict(raw)
    if verdict not in {"supported", "partial", "unsupported"}:
        logger.warning("grounding_check got unparseable response %r, failing open", raw[:80])
        return JudgeVerdict(
            verdict="",
            reason=f"guard response unparseable: {raw[:80]}",
            failed_open=True,
            verified=False,
        )
    return JudgeVerdict(verdict=verdict)


def is_unsupported(verdict: JudgeVerdict) -> bool:
    # only an explicit UNSUPPORTED trips a refusal; "" (no verdict) passes
    return verdict.verdict == "unsupported"


def ungrounded_refusal() -> str:
    return UNGROUNDED_REFUSAL
