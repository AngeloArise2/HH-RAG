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

Live at: **https://voice-rag-j9oh.onrender.com** — permanent home on **Render free tier**, deployed from this repo's `render.yaml` blueprint + root `Dockerfile` (single service: FastAPI backend + built frontend served same-origin).

Verified live (Aug 22, 2026):

```
$ curl https://voice-rag-j9oh.onrender.com/health
{"status":"ok"}
```

Real query through the deployed backend (`POST /ask`, "who owns a corporation and who shares in its profits?"): grounded answer returned — *"A corporation is owned by its stockholders (shareholders), and those shareholders share in the corporation's profits."* — with `grounding_verified: true`. Off-topic probe ("what time is it right now") correctly refused (`refusal_reason: off_topic`) with only `guardrail_check` in the trace (278ms) — the input guard short-circuits before any embedding runs. During a 20-query benchmark against this deployment, guardrails refused 7 more real dataset queries: 2 unsafe, 4 ungrounded, 1 off-topic.

**Deployed latency:** retrieval-only P50/P70/P100 on this deployment is **10.0 / 12.0 / 17.3ms** — inside the 200ms budget, measured across all three retrieval-budget stages (embed_query + vector_search + chunk_assembly) against the live URL with real dataset queries (`scripts/bench_remote.py`; full diagnostic arc in `docs/latency_report.md`). Getting here took three real fixes: a per-request Chroma collection reopen, ONNX thread pools auto-sized to host cores on a 0.1-CPU container, and a chunk-assembly stage that was documented but never actually timed — each found by questioning the numbers rather than accepting them. End-to-end (including LLM generation + grounding judge) runs ~750ms p50 deployed and carries no 200ms claim.

Platform history — measured, not guessed:

1. **Render free — originally rejected.** With the torch-based embedding runtime the container idled at ~483MB *anonymous* memory (verified inside the built image; malloc/thread mitigations moved it <2%) against Render's 512MB cap → guaranteed OOM before the first request. We bridged to a Railway trial.
2. **ONNX swap changed the math.** Embedding inference moved to ONNX Runtime (same all-MiniLM-L6-v2 weights, parity-gated at cosine = 1.000000 over real corpus samples — see `docs/architecture.md`). Measured idle anon dropped to **283MB** and the image from 3.04GB → 1.61GB.
3. **Render free — redeployed, now permanent.** Memory fits with ~45% headroom. The Railway trial service is decommissioned once Render confirms stable (see `docs/submission_checklist.md`).

Deploy steps (what was actually done):

```bash
# 0) index snapshot must be committed (it ships INSIDE the docker build,
#    so deploys never download the dataset or re-embed):
#    .venv/bin/python scripts/export_index_snapshot.py   # → backend/data/index_snapshot.tgz (~33MB)

# 1) push the repo (Render deploys from GitHub):
git push origin main

# 2) dashboard.render.com → New → Blueprint → select AngeloArise2/HH-RAG;
#    render.yaml pre-configures service name, Docker runtime, free plan,
#    /health check. Paste GROQ_API_KEY + SARVAM_API_KEY when prompted
#    (sync:false → values live only in Render's vault).

# 3) first build ~5-8 min; then verify:
curl https://<service>.onrender.com/health   # expect {"status":"ok"}
```

Free-tier caveats, documented honestly: (1) the service spins down after ~15min idle; the next request pays a ~50-60s cold boot — hit `/health` once before demoing or judging. (2) Generation/guardrail stages track Groq's network latency (~165-508ms p50); the retrieval budget itself is unaffected.

The frontend is served by the backend itself from `frontend/dist` (same-origin, single service) — record voice or type a question at the domain root.


## Team & workflow

Built using OpenCode inside VS Code, following the phased plan in `BUILD_PROMPT.md`. See `AGENTS.md` for engineering conventions and `REVIEW.md` for the review process and final submission checklist.

## Submission

See `REVIEW.md`'s final checklist — form link, GitHub link, live link, two videos, and the mandatory Instagram + X promotion requirements (`#RAGInGoa`, every team member, at least one public Instagram account). **No resubmissions allowed** — double-check before submitting.