# OPENCODE_SKILLS.md

This documents the OpenCode-specific tooling in this repo — what's already set up in `.opencode/`, how it works, and how to extend it. This is different from `SKILLS.md` (that one's a glossary for you, the human; this one's about OpenCode's actual features).

OpenCode has two relevant extension mechanisms, and this project uses both:

## 1. Custom commands (`.opencode/commands/*.md`)

A markdown file in `.opencode/commands/` becomes a slash command — typing `/review-phase` in the OpenCode TUI sends that file's prompt to the agent. Already set up in this repo:

| Command | What it does | Agent mode |
|---|---|---|
| `/review-phase` | Reviews the most recent phase's code against `BUILD_PROMPT.md`/`REVIEW.md` acceptance criteria. **Read-only** — flags issues, doesn't fix them. | `plan` (read-only), runs as an isolated subtask so it doesn't clutter your main conversation |
| `/explain-last` | Explains the last change in plain English, using the vocabulary from `SKILLS.md`. For you, not for grading. | `plan`, subtask |
| `/latency-check` | Runs `scripts/run_benchmark.py` and reports P50/P70/P100 against the 200ms retrieval target, honestly separated from full end-to-end. | `build` (needs to actually run code) |
| `/guardrail-test` | Runs the guardrail adversarial test suite and reports pass/fail per category, including false refusals. | `build` |

Each file has YAML frontmatter (`description`, `agent`, `subtask`) followed by the prompt template. Templates can pull in live shell output with `` !`command` `` and reference files with `@path/to/file` — both are used here (e.g. `/review-phase` injects `git diff --stat` automatically so you don't have to paste it).

**How to use them:** after pasting a phase from `BUILD_PROMPT.md` and letting the agent finish, run `/review-phase` before you commit. If it comes back clean, run `/explain-last` to actually understand what got built, *then* commit. Don't skip straight to the next phase on a "looks done" vibe.

## 2. Agent Skills (`.opencode/skills/<name>/SKILL.md`)

Skills are reusable playbooks the agent loads **on demand** — it sees a list of available skill names/descriptions and decides to pull one in when relevant, rather than you having to re-explain conventions every session. Already set up:

- **`chunking-strategy`** — the rules for implementing/extending chunkers (interface, required strategies, test expectations). Auto-relevant whenever the agent touches `backend/app/chunking/`.
- **`latency-instrumentation`** — the rules for timing pipeline stages and computing P50/P70/P100 honestly. Auto-relevant whenever touching `backend/app/benchmarking/` or the benchmark script.
- **`rag-guardrails`** — the rules for implementing and wiring guardrails into the actual request path (not just writing a function nobody calls). Auto-relevant whenever touching `backend/app/guardrails/` or the orchestrator's guardrail calls.

You don't invoke these directly — the agent calls the `skill` tool itself when a task matches. You can force the point by mentioning the skill name in a phase prompt if the agent seems to be skipping the convention (e.g. "follow the chunking-strategy skill for this").

## 3. `AGENTS.md` — the standing rules file

This is OpenCode's native project-rules file (equivalent to Cursor's `.cursorrules`). It's loaded into context automatically every session — this is where the non-negotiable engineering rules live (timing every retrieval stage, no naive-only chunking, guardrails wired into the request path, etc.). You don't need to repeat these in every `BUILD_PROMPT.md` phase; they're always in context. If you ever run `/init`, OpenCode will try to merge/update `AGENTS.md` from scanning the repo — that's fine, but review the diff, since our version has task-specific rules a generic scan won't infer.

## Adding a new command or skill mid-project

- **New command:** drop a `.md` file in `.opencode/commands/`, filename becomes the command name. Copy the frontmatter shape from an existing one.
- **New skill:** `mkdir .opencode/skills/<kebab-case-name>` and add a `SKILL.md` with `name` + `description` frontmatter (name must match the folder name exactly, lowercase, hyphen-separated). Keep the description specific — it's the only thing the agent sees before deciding to load the full skill.

Both are picked up automatically — no restart needed, no registration step beyond creating the file.
