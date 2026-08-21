---
description: Review the most recent phase's code against BUILD_PROMPT.md and REVIEW.md acceptance criteria, without editing anything
agent: plan
subtask: true
---

Recent changes:

!`git log --oneline -5`

!`git diff HEAD~1 --stat`

You are reviewing, not implementing. Read @BUILD_PROMPT.md and @REVIEW.md to find the phase that was just worked on (match it against the files touched above), then:

1. List the acceptance criteria for that phase from `REVIEW.md`.
2. For each criterion, state pass / fail / partial, with a one-line reason pointing at the specific file or function.
3. Check the non-negotiable rules in `AGENTS.md` (timing instrumentation present, no naive-only chunking, guardrails actually wired into the request path, structured error handling, tests added) against what was actually written — don't take a docstring's word for it, look at the code.
4. Flag anything that looks like a stub, a TODO, a hardcoded fake value, or a mock silently standing in for a real call.
5. End with a short verdict: **ready to commit / needs fixes**, and if fixes are needed, a concrete numbered list of what to change — but do not make the changes yourself.

Do not modify files. This is a read-only review.

Extra argument (optional, a phase number or keyword to focus on): $ARGUMENTS
