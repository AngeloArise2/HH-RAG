# HH Goa 2026 — Voice-Enabled RAG

A voice-in, voice-question, grounded-answer-out RAG pipeline: **audio → speech-to-text → multi-strategy chunked retrieval over a vector DB → guardrailed answer generation**, built for the HH Goa 2026 shortlisting task on `ai4bharat/MSMARCO-XI`.

## Architecture

```
                  ┌─────────────┐
   voice clip ──▶ │  STT        │  (Sarvam or ElevenLabs)
                  └──────┬──────┘
                         │ transcript
                         ▼
                  ┌─────────────┐
                  │ Guardrail:  │  off-topic / unsafe input check
                  │ input filter│
                  └──────┬──────┘
                         │ (pass)
                         ▼
                  ┌─────────────┐     ┌───────────────────┐
                  │ Retriever   │────▶│ Vector DB (Chroma) │
                  │ embed+search│     │ 4 chunking-strategy│
                  └──────┬──────┘     │ collections        │
                         │ top-k chunks └───────────────────┘
                         ▼
                  ┌─────────────┐
                  │ Generation  │  LLM answers ONLY from retrieved context
                  └──────┬──────┘
                         │ answer
                         ▼
                  ┌─────────────┐
                  │ Guardrail:  │  groundedness / hallucination check
                  │ grounding   │
                  └──────┬──────┘
                         ▼
                  structured response
                  (answer / refusal / latency trace)
```

The whole flow above is `backend/app/harness/orchestrator.py` — every stage is timed, wrapped in retries/timeouts where it hits an external API, and returns a structured Pydantic response rather than a bare string.

## Repo structure

```
.
├── AGENTS.md                  # OpenCode's standing project rules (read this first if you're the agent)
├── README.md                  # you are here
├── PREREQUISITES.md           # Day 0 setup
├── BUILD_PROMPT.md            # phase-by-phase prompts to paste into OpenCode
├── REVIEW.md                  # phase checklist + final submission checklist
├── SKILLS.md                  # plain-English concept glossary (for humans)
├── OPENCODE_SKILLS.md         # how the .opencode/ commands & skills work
├── .opencode/
│   ├── commands/               # /review-phase, /explain-last, /latency-check, /guardrail-test
│   └── skills/                 # chunking-strategy, latency-instrumentation, rag-guardrails
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, /health, /ask, /transcribe
│   │   ├── config.py            # env-based settings
│   │   ├── stt/                 # Sarvam / ElevenLabs / mock providers
│   │   ├── ingestion/           # dataset download + preprocessing
│   │   ├── chunking/            # 4 chunking strategies + router
│   │   ├── retrieval/           # embeddings + Chroma vector store + retriever
│   │   ├── generation/          # LLM client + prompts
│   │   ├── guardrails/          # input filter, grounding check, refusal
│   │   ├── harness/             # orchestrator (the actual pipeline)
│   │   └── benchmarking/        # latency timing utilities
│   ├── tests/
│   └── data/                   # gitignored — downloaded dataset + built indexes
├── frontend/                   # minimal voice-recording UI
├── scripts/                    # build_index.py, run_benchmark.py, etc.
├── docs/
│   ├── architecture.md
│   ├── latency_report.md       # P50/P70/P100 numbers (Phase 8)
│   └── submission_checklist.md
└── videos/                      # local notes/footage, not committed raw
```

## Setup

See `PREREQUISITES.md` for the full Day 0 checklist. Quick version once `.env` is filled:

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

```bash
cd frontend
npm install
npm run dev
```

## Running the pipeline

```bash
# one-time: download + prepare dataset, build vector indexes
python scripts/download_and_prepare.py
python scripts/build_index.py

# run latency benchmark
python scripts/run_benchmark.py --queries 50
```

## Latency — read this before assuming a number is wrong

The task specifies a **sub-200ms** target for "chunking + vector DB retrieval + everything through to final output." We report this honestly, split two ways in `docs/latency_report.md`:

- **Retrieval-only** (chunk lookup + embedding + vector search) — this is what's checked against the 200ms target, and a local in-process vector DB genuinely can hit this.
- **Full end-to-end** (also including the STT API call and LLM generation) — reported separately and honestly. A live network call to an STT or LLM API routinely exceeds 200ms on its own; we're not going to misreport that number to look compliant. See `SKILLS.md`'s "Why 200ms is aggressive" section for the reasoning.

## STT provider

This project uses **[Sarvam / ElevenLabs — fill in once decided, see PREREQUISITES.md]**, per the task's "pick one" requirement. Both providers are implemented behind a common interface in `backend/app/stt/` for flexibility during development, but only the configured one (`STT_PROVIDER` in `.env`) is used at runtime.

## Deployment

Backend: **[fill in once deployed — Render / Railway / Fly.io]** → live URL: **[fill in]**
Frontend: **[fill in if separately deployed]**

## Team & workflow

Built using OpenCode inside VS Code, following the phased plan in `BUILD_PROMPT.md`. See `AGENTS.md` for engineering conventions and `REVIEW.md` for the review process and final submission checklist.

## Submission

See `REVIEW.md`'s final checklist — form link, GitHub link, live link, two videos, and the mandatory Instagram + X promotion requirements (`#RAGInGoa`, every team member, at least one public Instagram account). **No resubmissions allowed** — double-check before submitting.
