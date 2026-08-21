---
title: voice-rag
emoji: 🎙️
sdk: docker
app_port: 7860
---

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

This project uses **Sarvam** (`STT_PROVIDER=sarvam` in `.env`), per the task's "pick one" requirement. Both Sarvam and ElevenLabs are implemented behind a common interface in `backend/app/stt/` for flexibility, but only the configured one is used at runtime.

## LLM provider

Answer generation uses **Groq** (`openai/gpt-oss-20b` — the free-tier fast general model as of Aug 2026) with **Gemini** as a config-swap fallback if rate limits get tight. Both live behind one interface in `backend/app/generation/llm_client.py`.

## Deployment

Live at: **[fill in real Space URL after deploy]** — deployed as a Hugging Face Space running this repo's `Dockerfile` verbatim.

Why HF Spaces and not Render/Railway/Fly (measured, not guessed): the container idles at ~480MB *anonymous* memory (torch runtime + MiniLM + interpreter — verified inside the built image; `MALLOC_ARENA_MAX`/thread-cap mitigations moved it <2%). Render's free tier caps at 512MB total → guaranteed OOM before the first request. Spaces' free CPU tier provides 16GB, no card required, same image. Tradeoffs accepted and documented: the Space sleeps after ~48h idle (first visit pays a ~1min cold boot: container start + MiniLM warmup) — hit `/health` once before demoing.

Deploy steps (what was actually done):

```bash
# 0) index snapshot must be committed (it ships INSIDE the docker build,
#    so deploys never download the dataset or re-embed):
#    .venv/bin/python scripts/export_index_snapshot.py   # → backend/data/index_snapshot.tgz (~33MB)

# 1) one-time human steps: huggingface.co account → Settings → Access Tokens →
#    create a WRITE token; hf.co/new-space → name "voice-rag", SDK=Docker, public

# 2) push the repo to the Space
git remote add space https://huggingface.co/spaces/<your-username>/voice-rag
git push space main        # token = password when prompted

# 3) Space → Settings → Variables and secrets → add GROQ_API_KEY, SARVAM_API_KEY

# 4) watch the build logs (first build ~6-8 min: CPU torch + deps + weights bake),
#    then verify:
curl https://<your-space-subdomain>.hf.space/health
```

The frontend is served by the backend itself from `frontend/dist` (same-origin, single service) — record voice or type a question at the Space URL root.


## Team & workflow

Built using OpenCode inside VS Code, following the phased plan in `BUILD_PROMPT.md`. See `AGENTS.md` for engineering conventions and `REVIEW.md` for the review process and final submission checklist.

## Submission

See `REVIEW.md`'s final checklist — form link, GitHub link, live link, two videos, and the mandatory Instagram + X promotion requirements (`#RAGInGoa`, every team member, at least one public Instagram account). **No resubmissions allowed** — double-check before submitting.