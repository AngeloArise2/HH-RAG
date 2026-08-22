# AGENTS.md

This file is read automatically by OpenCode on every session — it's the project's standing instructions. Commit it to git. Don't duplicate this content into every phase prompt; the agent already has it.

## What this project is

A voice-enabled RAG pipeline: **audio in → speech-to-text → multi-strategy chunking/retrieval over a vector DB → grounded answer out**, built for the HH Goa 2026 shortlisting task. Dataset: `ai4bharat/MSMARCO-XI` (Hugging Face). Hard constraints: retrieval-side latency under 200ms (P50/P70/P100 reported), a real orchestration harness (not a bare prompt call), and guardrails that make the system refuse when it should.

## Stack (do not silently change these — if a phase seems to require a different tool, stop and ask instead of swapping)

- **Backend:** Python 3.10+, FastAPI
- **STT:** whichever of Sarvam / ElevenLabs is set in `.env` as `STT_PROVIDER` — code must support both behind one interface even though only one is "live," so switching is a config change, not a rewrite
- **Vector store:** Chroma (in-process, local disk) — chosen for latency, not because it's the only correct choice. Don't switch to a hosted vector DB.
- **Embeddings:** all-MiniLM-L6-v2 weights, served through **ONNX Runtime** (swapped from torch/sentence-transformers in Aug 2026 for a ~170MB idle-RAM cut, parity-gated — see docs/architecture.md). No network round-trip in the hot path. Only use an API-based embedder if a phase explicitly says so.
- **LLM generation:** whichever of Groq / Gemini is set as `LLM_PROVIDER` — Groq is the default (free tier, no card, and fast enough that it doesn't fight the latency budget); Gemini is the fallback if Groq's rate limits get hit during heavy testing or a live demo
- **Frontend:** plain Vite + vanilla JS/React, minimal — this is not the part being graded on polish
- **Tests:** `pytest` for backend

## Directory structure (see README.md for the diagram — keep new files inside this shape, don't invent new top-level folders without saying so)

```
backend/app/{stt,ingestion,chunking,retrieval,generation,guardrails,harness,benchmarking}
backend/tests/
backend/data/            # gitignored — downloaded dataset + built indexes live here
frontend/src/
scripts/                 # one-off / repeatable ops scripts (build_index.py, run_benchmark.py)
docs/                    # architecture.md, latency_report.md, submission_checklist.md
```

## Non-negotiable engineering rules

1. **Every retrieval-path function that touches STT, chunking, embedding, or vector search must be timed.** Use `backend/app/benchmarking/latency.py`'s timer utilities — don't hand-roll `time.time()` calls scattered around; centralize it so the benchmark script can read consistent timing data.
2. **The 200ms target applies to chunking + vector DB retrieval + everything through to final output, per the task doc — but do not fudge this number.** If full end-to-end (including LLM generation) can't hit 200ms, report the retrieval-only number honestly as the number that meets spec, and report full end-to-end separately and honestly in `docs/latency_report.md`. Never drop a stage from timing to make a number look better.
3. **No single naive fixed-size chunker as the only strategy.** At minimum: fixed-size w/ overlap, semantic/sentence-boundary-aware, and metadata-aware chunking must all exist behind a common interface (`backend/app/chunking/router.py`), even if only one is default at query time.
4. **Guardrails are not an afterthought bolted onto Phase 7 with no teeth.** The orchestrator (harness) must actually call the guardrail checks in the request path — off-topic detection, unsafe-input detection, and a groundedness check that compares the answer against retrieved context before returning it.
5. **The harness must have real error handling:** timeouts on external calls (STT API, LLM API), at least one retry with backoff on transient failures, and a structured (Pydantic) response schema — not a dict grab-bag.
6. **Write a test for every new module** in `backend/tests/`, even a thin one. If a phase adds a chunker, add a test asserting basic properties (chunk count > 0, no empty chunks, overlap behaves as configured).
7. **Update `REVIEW.md`'s phase checklist** (check off completed acceptance criteria) as part of finishing a phase, not as a separate chore later.
8. **Commit at the end of every phase** with a message like `phase 3: embedding + vector index`. Small, phase-scoped commits — don't squash weeks of work into one commit at 11:58 PM.

## Working style

- Prefer editing existing stub files created in Phase 0 over creating new ones with slightly different names.
- If a task in `BUILD_PROMPT.md` is ambiguous, make the most reasonable engineering choice, note the assumption in a code comment or in `docs/architecture.md`, and keep moving — this project is on a one-day clock, don't stall on clarification.
- Keep explanations of *why* (not just *what*) in commit messages and docstrings; the human running this project is learning these concepts as they go (see `SKILLS.md`) and will use `/explain-last` to understand what happened.
- Never fabricate benchmark numbers, dataset stats, or claims of what was tested. If something wasn't run, say so.

## Commands available in this repo

See `.opencode/commands/` — `/review-phase`, `/explain-last`, `/latency-check`, `/guardrail-test`. Documented in `REVIEW.md`.