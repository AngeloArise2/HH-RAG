"""Refusal responses users actually see when a guardrail trips.

Rules for these strings:
- Plain language, explain WHY, never a bare empty result.
- The off-topic/grounded refusals name the dataset grounding so the limit
  reads as a design decision, not a bug.
- UNSAFE refusal does not repeat or paraphrase the offending content.
"""

OFF_TOPIC_REFUSAL = (
    "I couldn't find grounded information about that in the dataset this "
    "system searches, so I won't guess. Try asking about a topic you'd expect "
    "in general reference passages (business, history, science, everyday life)."
)

UNSAFE_REFUSAL = (
    "I can't help with that request. If you have a different question, "
    "I'm happy to try answering it from the retrieved passages."
)

UNGROUNDED_REFUSAL = (
    "I drafted an answer, but I couldn't verify it against the retrieved "
    "passages, so I'm not sharing it rather than risk telling you something "
    "unsupported. Try rephrasing the question."
)


def refusal_for(verdict: str) -> str:
    """Map a guard verdict to its user-facing refusal."""
    return {
        "off_topic": OFF_TOPIC_REFUSAL,
        "unsafe": UNSAFE_REFUSAL,
        "ungrounded": UNGROUNDED_REFUSAL,
    }.get(verdict, OFF_TOPIC_REFUSAL)
