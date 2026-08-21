---
name: rag-guardrails
description: Playbook for implementing and testing guardrails (off-topic detection, unsafe-input handling, groundedness/hallucination checks) for this project's RAG pipeline. Use whenever touching backend/app/guardrails/ or backend/app/harness/orchestrator.py's guardrail calls.
license: MIT
compatibility: opencode
---

## What I do

Keep guardrails wired into the actual request path — not a separate demo function nobody calls.

## The three checks

1. **`input_filter.py`** — runs on the raw (transcribed) query *before* retrieval. Two jobs: (a) off-topic detection — is this question plausibly answerable from the MSMARCO-XI dataset domain, cheaply, e.g. via a lightweight classifier or a fast LLM call with a strict yes/no schema, not a full generation; (b) unsafe/inappropriate input detection — refuse before spending a retrieval call on it.
2. **`grounding_check.py`** — runs *after* generation, *before* returning the answer. Compares the generated answer against the retrieved chunks (e.g. via NLI-style entailment check, or an LLM-as-judge call constrained to a structured yes/no/partial verdict) and flags answers not supported by the context.
3. **`refusal.py`** — the actual refusal response the user sees when either check trips — should explain *why* in plain terms ("I couldn't find grounded information about that in the dataset") not just return an empty result.

## Wiring rule (this is the part that's easy to fake)

`backend/app/harness/orchestrator.py` must call `input_filter` before retrieval and `grounding_check` after generation, in the actual request flow — verify this by checking the orchestrator's code path, not by reading the guardrail module's docstring. `/review-phase` checks this specifically.

## Test set

`backend/tests/test_guardrails.py` needs at least:
- 3+ off-topic queries that should be refused
- 3+ in-scope, perfectly normal queries that should **not** be refused (false-refusal check — a system that blocks everything isn't a guardrail, it's broken)
- 2+ adversarial/unsafe inputs
- 2+ cases engineered to produce a plausible-sounding but ungrounded answer, to verify the grounding check actually catches something

`/guardrail-test` runs this suite and reports per-category results, including false refusals.
