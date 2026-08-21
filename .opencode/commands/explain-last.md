---
description: Explain the last phase's changes in plain English for a non-technical teammate
agent: plan
subtask: true
---

Recent changes:

!`git log --oneline -5`

!`git diff HEAD~1`

The person reading this barely knows these concepts yet — see @SKILLS.md for the vocabulary they're comfortable with; stick to those terms and define anything new the same way that file does.

Explain what just happened, in this shape:

1. **In one sentence:** what did this phase accomplish, in a real-world analogy if it helps.
2. **What files changed and why**, in plain language — not a line-by-line diff walkthrough, a "this file is responsible for X, and we taught it to also do Y" summary.
3. **What can the project do now that it couldn't do before this phase?** Be concrete — e.g. "you can now record a voice question and get back a text transcript" rather than "STT integration complete."
4. **What's still missing** before this piece is actually useful end-to-end (e.g. "transcription works, but nothing uses that transcript yet — that's the next phase").
5. Skip code snippets entirely unless one specific line is genuinely necessary to make a point — then explain it in words immediately after.

No jargon without a one-clause definition attached the first time it's used. No code review, no opinions on quality — that's what `/review-phase` is for.

Optional focus argument: $ARGUMENTS
