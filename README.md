# HH Goa 2026 — Voice-Enabled RAG

A voice-in, grounded-answer-out RAG pipeline: **audio → speech-to-text → multi-strategy chunked retrieval over a vector DB → guardrailed answer generation**, built on `ai4bharat/MSMARCO-XI`.

**Live:** https://voice-rag-j9oh.onrender.com · **Retrieval P50/P70/P100 deployed: 10.0 / 12.0 / 17.3 ms** (target < 200 ms) · **140 tests across 18 files**

---

## Engineering at a glance

| Area | What we actually built | Evidence |
|---|---|---|
| Chunking | 4 strategies (fixed-size, fixed-size+overlap, metadata-aware, semantic) behind one `Chunker` interface + router | `backend/app/chunking/`, per-strategy Chroma collections |
| Embeddings | torch → **ONNX Runtime** swap, same weights, **parity-gated at cosine ≥ 0.999 (measured 1.000000)** | `backend/tests/test_embedding_parity.py` |
| Latency | Centralized per-stage timers covering all three retrieval-budget stages, P50/P70/P100, honest retrieval-vs-e2e split | `backend/app/benchmarking/latency.py`, `docs/latency_report.md` |
| Guardrails | Input filter (off-topic/unsafe) **short-circuits before retrieval**; grounding judge blocks fabricated answers; structured fail-open contract | `backend/app/guardrails/`, orchestrator call sites |
| Orchestration | Timeouts + retry/backoff on every hosted API call, Pydantic response schema, per-stage trace in every response | `backend/app/harness/orchestrator.py` |
| Deployed perf | Debugged an 824 ms p50 misread as "hardware ceiling" down to **three real defects → 10.0 ms p50**, >10× headroom at P100 | commits `2f66af8`, `af463e4`; tables below |
| Testing | 140 tests across 18 pytest files, including cross-runtime embedding parity | `backend/tests/` |

---

## Architecture

```
                  ┌─────────────┐
   voice clip ──▶ │  STT        │  (Sarvam or ElevenLabs, one interface)
                  └──────┬──────┘
                         │ transcript
                         ▼
                  ┌─────────────┐
                  │ Guardrail:  │  off-topic / unsafe input check
                  │ input filter│  (refuses BEFORE any embedding runs)
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
                  │ Guardrail:  │  groundedness / hallucination judge
                  │ grounding   │  (fail-open, flagged — never silent)
                  └──────┬──────┘
                         ▼
                  structured Pydantic response
                  (answer / refusal / warnings / latency trace)
```

Every stage above lives in `backend/app/harness/orchestrator.py`, is timed via shared benchmarking utilities (no hand-rolled `time.time()`), and external calls carry timeouts and at least one retry with backoff.

## The engineering, in detail

### 1. Retrieval path built for the 200 ms budget

The task's 200 ms target physically cannot include mandated hosted APIs (Sarvam/ElevenLabs round-trip alone measures ~1.3 s), so we report two numbers and never blur them:

- **Retrieval-only** (chunking → embedding → vector search → assembly): the number checked against 200 ms. Local p50 **9.3 ms** (n=50); deployed p50 **10.0 ms**.
- **Full end-to-end** (incl. generation + both LLM guard calls): reported separately with no 200 ms claim attached. Local text-mode p50 1713 ms; deployed total p50 ~750 ms.

Methodology choices that make the numbers trustworthy: queries taken from the dataset's own `Eng_Query` column (not hand-invented inputs); embedder warmup plus one discarded pipeline call absorbed into FastAPI `lifespan` so no user pays cold-start (measured 9–11.5 s cold); runs paced to avoid Groq TPM 429-retries polluting e2e samples; refusals counted as refusals, not silently excluded.

### 2. Four chunking strategies, one interface

`fixed_size`, `fixed_size_overlap`, `metadata_aware`, and `semantic` all implement the same `Chunker` protocol behind `chunking/router.py`; the retriever selects a strategy-built collection by config. Stable ids (`sha1(cleaned_text)[:16]`) make passages dedupe across query rows and survive re-runs. Strategy choice is config, not code.

### 3. Embeddings: swapped inference engine, kept the model — with a parity gate

Post-deploy memory measurement showed the container idling at ~483 MB anonymous RSS, almost entirely the torch runtime sentence-transformers imports (Render free caps at 512 MB). We swapped the *runtime*, not the model:

- Official `model.onnx` weights from the HF repo (90 MB) + `transformers` tokenizer; mean-pool + L2-normalize replicated exactly from model config.
- `sentence-transformers`/torch removed from requirements entirely (ST v6 imports torch eagerly even with its ONNX backend — keeping it would have kept the 480 MB).
- **Gate, not hope:** `test_embedding_parity.py` pushes ~21 real corpus passages through both runtimes and asserts per-sample cosine ≥ 0.999. Measured minimum: **1.000000**.

Measured effect: idle anon RSS 483 → 283 MB (−41%), Docker image 3.04 → 1.61 GB, `embed_query` p50 ~14 → ~6 ms locally, 123.8 → **~6 ms** deployed.

### 4. Guardrails wired into the request path — with a tested failure contract

Both guards run inside the orchestrator, not bolted on afterward:

- **Input filter** (before retrieval): off-topic + unsafe detection; refuses before any embedding work. In the local benchmark it refused 12/40 real queries (4 off-topic, 3 unsafe — e.g. overdose and piracy how-tos).
- **Grounding judge** (after generation): an LLM compares the answer against retrieved context; fabricated answers never reach the caller (caught 5/40 in the local benchmark).
- **Structured fail-open, surfaced:** if the judge's API call fails (Groq 429/timeouts observed live), the answer still returns with `grounding_verified=false` plus the exception in `warnings[]` — availability-over-blocking, by explicit design, documented in `docs/architecture.md`. `refused=true` (a guard blocked) and `grounding_verified=false` (returned but unjudged) are independent fields covering different failure modes, with tests pinning that distinction.
- Known edge cases are published, not hidden: a few false-refusal candidates and judge fail-opens are logged in the latency report for tuning.

### 5. Deployed performance: the 824 ms that wasn't hardware

First deployed benchmark read **824 ms p50** and the initial write-up blamed CPU allocation. Reviewer pushback (embed 124 ms vs search ~700 ms scaling *backwards*) pointed instead at per-request work. Three real defects found and fixed:

1. **Chroma collection handle reopened every request** — fresh `PersistentClient`, handle fetch, and two `collection.count()` sqlite round-trips per query (`2f66af8`). Fixed: handles cached per `(store_path, name)`, warmed once at startup.
2. **ORT thread pools auto-sized from host cores** on a container with 0.1 shared CPU — pure contention (`2f66af8`). Fixed: pinned `intra_op_num_threads = inter_op_num_threads = 1`.
3. **A retrieval stage that was documented but never timed** — reviewer question "requirement 3 includes chunking; where is it in this table?" exposed that `CHUNK_ASSEMBLY` existed as a defined stage but no request-path code ever started its timer; totals silently summed only the stages present (`af463e4`). Fixed: query-time chunk rendering now records the timer through the ambient request trace.

Result, measured against the live URL with real dataset queries (`scripts/bench_remote.py`, server-side per-stage traces, 20 queries, n=13 after 7 guard refusals):

| stage | first deploy p50 | **final p50** | final p70 | final p100 |
|---|---:|---:|---:|---:|
| embed_query | 123.8 | **5.8** | 7.0 | 9.5 |
| vector_search | 698.5 | **4.0** | 5.0 | 8.2 |
| chunk_assembly | untimed | **~0.02** | ~0.02 | ~0.05 |
| **retrieval_ms** | 824.0 | **10.0** | 12.0 | 17.3 |
| generation | 298.1 | 194.4 | 282.5 | 505.3 |
| guardrail_check | 442.3 | 481.1 | 586.2 | 1899.2 |
| total_ms | 1569.4 | 750.3 | 872.3 | 2183.1 |

(Interim post-fix run with only two stages wired read 14.8/16.9/27.1 ms — kept for provenance.)

Two lessons recorded in the repo: "hardware ceiling" is a diagnosis you earn by ruling out per-request setup first — and a stage that isn't timed is a stage that quietly isn't in your budget.

### 6. Latency-optimization judgment: evaluated, not ignored

Every candidate pattern got an explicit adopt/reject decision with reasoning (`docs/latency_report.md`):

| pattern | decision | reason |
|---|---|---|
| Startup warmup (embedder + discarded first pipeline call) | **adopted** | kills the 9–11.5 s cold-start spike at zero steady-state cost |
| Generation token ceiling 512 → 384 | **adopted** | trims decode tail; stays above gpt-oss's empty-content cliff seen with 16-token caps |
| Speculative parallel guard ∥ retrieval | rejected | saves only ~25 ms locally — embedding is in-process, not cloud |
| Semantic cache | rejected | benchmarks grade distinct queries; cited "110 ms → 0.35 ms" results elsewhere are cache hits |
| Streaming STT / async mid-stream guards | rejected / deferred | needs WebSocket/SSE re-architecture late on a one-day clock; final-answer quality needs the full transcript anyway |
| Swap Chroma → HNSWlib/Qdrant | rejected | vector search p70 already ~5 ms deployed; violates fixed stack for milliseconds we don't need |
| Self-hosted LLM | rejected | CPU decode ≈10–25 tok/s vs Groq ≈1000 tok/s × three LLM calls/request → est. 10× slower e2e |

### 7. Dataset engineering

MSMARCO-XI ships no flat corpus — rows are queries with nested passage lists (~55 GB total). We flatten `English_passages` one-row-to-N-documents, cap ingestion inside the 5k–20k passage spec window, and preserve provenance metadata (`query_id`, `is_selected`, `query_type`, language, split) end-to-end — `is_selected` doubles as free relevance labels for future eval work without any manual labeling.

### 8. Provider abstraction where requirements demand it

STT: Sarvam and ElevenLabs implemented behind one interface (`stt/base.py` + factory) — switching is one `.env` change, verified with mock-provider tests offline. LLM: Groq (`openai/gpt-oss-20b`) primary, Gemini fallback, same pattern. Only the configured provider ever loads at runtime.

### 9. Deployment engineering

- Single Docker service on Render free tier: FastAPI serves the built Vite frontend same-origin; `render.yaml` blueprint, secrets via `sync:false` vault entries.
- The ~33 MB vector-index snapshot ships **inside the image** (exported via `scripts/export_index_snapshot.py`) — deploys never download the dataset or re-embed.
- Free-tier physics documented up front: ~15 min idle spin-down, ~50–60 s cold boot (hit `/health` before demoing); generation latency tracks Groq's network position (~194–481 ms p50), not ours.
- Deployment history kept honest: torch-era memory made Render free untenable → Railway trial bridge → ONNX swap changed the math → back to Render permanent with ~45% memory headroom.

---

## Measured results (final)

Retrieval-only vs target: deployed **P50 10.0 / P70 12.0 / P100 17.3 ms** across all three budget stages — meets 200 ms with >10× headroom at P100; local three-stage equivalent 9.3 / 9.8 / 12.3 ms (n=50). Full diagnostic arc, per-stage percentiles, chunking-strategy comparison, and guardrail trip tables: [`docs/latency_report.md`](docs/latency_report.md).

## Repo structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, /health, /ask, /transcribe
│   │   ├── stt/                 # Sarvam / ElevenLabs / mock behind one interface
│   │   ├── ingestion/           # MSMARCO-XI download + preprocessing
│   │   ├── chunking/            # 4 strategies + router
│   │   ├── retrieval/           # ONNX embeddings + Chroma + retriever
│   │   ├── generation/          # LLM client + prompts (Groq/Gemini)
│   │   ├── guardrails/          # input filter, grounding judge, refusal
│   │   ├── harness/             # orchestrator: timeouts, retries, tracing
│   │   └── benchmarking/        # centralized latency utilities
│   ├── tests/                   # 140 tests, 18 files
│   └── data/                    # gitignored — dataset + indexes (+ committed index_snapshot.tgz)
├── frontend/                    # minimal voice UI (Vite, vanilla JS)
├── scripts/                     # build_index.py, run_benchmark.py, bench_remote.py, ...
├── docs/
│   ├── architecture.md          # design decisions & rationale
│   ├── latency_report.md        # P50/P70/P100 + methodology + decisions
│   └── submission_checklist.md
├── render.yaml + Dockerfile     # deploy blueprint
└── AGENTS.md / REVIEW.md / BUILD_PROMPT.md / SKILLS.md
```

## Running it

```bash
cd backend && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ../scripts/download_and_prepare.py   # one-time: fetch + flatten dataset
python ../scripts/build_index.py            # one-time: embed + build 4 collections
uvicorn app.main:app --reload

# frontend
cd frontend && npm install && npm run dev

# reproduce the benchmark
.venv/bin/python scripts/run_benchmark.py --queries 40 --sleep 15 --output docs/latency_report.md
.venv/bin/python scripts/bench_remote.py    # same harness, against the live deployment
```

Providers are set via `.env`: `STT_PROVIDER=sarvam|elevenlabs`, `LLM_PROVIDER=groq|gemini`. See `PREREQUISITES.md` for keys.

## Team & workflow

Built with OpenCode in phased commits (`phase N:` history in git log), phase-scoped reviews tracked in `REVIEW.md`, engineering rules enforced by `AGENTS.md` (timed stages, no fudged numbers, tests per module). See `SKILLS.md` for the concept glossary behind the design decisions.


