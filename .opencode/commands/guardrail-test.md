---
description: Run the guardrail adversarial test suite and summarize which cases pass
agent: build
---

Run the guardrail tests:

!`pytest backend/tests -k guardrail -v 2>&1 | tail -80`

Summarize the results:

1. How many adversarial cases were tested, grouped by category (off-topic, unsafe input, ungrounded/hallucinated answer, normal in-scope question that should NOT be refused).
2. Pass/fail per category, not just an aggregate count.
3. **False refusals matter as much as missed guardrails** — flag any case where a perfectly reasonable in-scope question got incorrectly refused, since a system that refuses everything technically "passes" guardrail tests but fails the actual task.
4. If the test file doesn't exist yet, say so and point at Phase 7 in `BUILD_PROMPT.md`.

Do not weaken or delete a failing test to make this pass — report the failure.
