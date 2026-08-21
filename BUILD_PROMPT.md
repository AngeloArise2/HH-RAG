# BUILD_PROMPT.md

Ten phases. Paste **one phase's prompt at a time** into OpenCode (as a new message in the same session is fine — `AGENTS.md` is always in context, so you don't need to repeat conventions). After each phase:

1. Let the agent finish.
2. Run `/review-phase`.
3. Fix anything flagged (paste the fix list back to the agent, or ask it to address its own review).
4. Run `/explain-last` if you want to actually understand what happened.
5. `git add . && git commit -m "phase N: <name>"`.
6. Check off the phase's boxes in `REVIEW.md`.
7. Move to the next phase.

Phases 2–3 and 5 can run in parallel on separate branches if your team is splitting work (chunking/retrieval vs. STT are independent until Phase 6 wires them together). Everyone should still do Phase 0 and 1 on `main` first.

---

## Phase 0 — Scaffold

```
Set up the project skeleton per AGENTS.md's directory structure. Specifically:

1. backend/: FastAPI app in backend/app/main.py with a single GET /health endpoint returning {"status": "ok"}. Set up backend/app/config.py loading settings from .env via pydantic-settings (STT_PROVIDER, SARVAM_API_KEY, ELEVENLABS_API_KEY, LLM_PROVIDER, GROQ_API_KEY, GEMINI_API_KEY, GROQ_MODEL, VECTOR_STORE_PATH).
2. Create backend/requirements.txt with: fastapi, uvicorn, pydantic-settings, pytest, httpx, chromadb, sentence-transformers, python-multipart, tenacity (for retries).
3. Empty __init__.py files so backend/app and its subpackages (stt, ingestion, chunking, retrieval, generation, guardrails, harness, benchmarking) are importable packages.
4. A minimal backend/tests/test_health.py using FastAPI's TestClient asserting GET /health returns 200.
5. .gitignore covering .env, backend/data/*, __pycache__, .venv, node_modules, *.pyc.
6. Confirm the app runs: uvicorn app.main:app --reload from backend/, and the health test passes.

Do not touch chunking, retrieval, STT, or generation logic yet — this phase is purely scaffolding and must end with a runnable, tested "hello world" backend.
```

**Acceptance:** `uvicorn` boots without error; `pytest backend/tests` passes; `git status` shows a clean, sensible file tree matching `AGENTS.md`.

---

## Phase 1 — Dataset ingestion

```
Implement dataset ingestion in backend/app/ingestion/:

1. download_dataset.py — downloads the ai4bharat/MSMARCO-XI dataset from Hugging Face (use the `datasets` library; add it to requirements.txt), saves a working subset (don't pull the entire dataset if it's huge — cap it via a config value, e.g. first 5,000-20,000 passages, enough to be a real retrieval corpus but fast to iterate on) to backend/data/raw/ as JSONL. Print basic stats when done: row count, avg passage length, a couple of sample rows.
2. preprocess.py — light cleaning (strip HTML/whitespace artifacts if any, drop empty/near-empty passages, assign a stable doc_id to each row) producing backend/data/processed/passages.jsonl.
3. A RawDocument pydantic model (id, text, metadata dict) shared by ingestion and chunking — put it in backend/app/ingestion/models.py so chunking (next phase) can import it.
4. scripts/download_and_prepare.sh (or .py) that runs both steps end to end.
5. Test asserting preprocess.py drops empty passages and preserves doc_id uniqueness.

Actually run the download and report real numbers (row count, sample passage) — don't guess at dataset shape.
```

**Acceptance:** `backend/data/processed/passages.jsonl` exists with real rows; the agent reports actual (not assumed) dataset statistics; test passes.

---

## Phase 2 — Multi-strategy chunking

```
Follow the chunking-strategy skill. Implement in backend/app/chunking/:

1. base.py — the Chunk pydantic model and Chunker protocol as specified in the skill.
2. fixed_size.py, fixed_size_overlap.py, semantic.py, metadata_aware.py — the four strategies.
3. router.py — a STRATEGIES registry (dict of name -> Chunker instance) and a get_chunker(name) function.
4. A test file per strategy in backend/tests/, following the skill's testing rules exactly.
5. scripts/build_chunks_preview.py — runs all four strategies against a small sample from backend/data/processed/passages.jsonl and prints chunk counts + one example chunk per strategy, so we can eyeball the difference between strategies.

Run the preview script and report actual output, not hypothetical examples.
```

**Acceptance:** four working chunkers behind one interface; tests pass; preview script shows visibly different chunking behavior across strategies on real data.

---

## Phase 3 — Embedding + vector index

```
Implement in backend/app/retrieval/:

1. embed.py — wraps a local sentence-transformers model (e.g. all-MiniLM-L6-v2 or similar small/fast model — pick one, note the choice and why in a comment) behind an embed_texts(list[str]) -> list[vector] function.
2. vector_store.py — wraps ChromaDB, persisted to VECTOR_STORE_PATH from config. Functions: build_index(chunks, strategy_name) creating a separate Chroma collection per chunking strategy, and query(text, strategy_name, top_k) returning matches with scores.
3. scripts/build_index.py — runs ingestion output through each of the four chunking strategies (Phase 2's router), embeds every chunk, and builds a Chroma collection per strategy. Report timing (how long indexing took) and collection sizes.
4. Test that build_index + query round-trips correctly on a tiny synthetic set of chunks (a query about topic X returns a chunk about topic X, not an unrelated one).

Actually run scripts/build_index.py against the real processed dataset from Phase 1 and report real numbers (collections built, chunk counts per strategy, total indexing time).
```

**Acceptance:** four persisted Chroma collections exist in `backend/data/index`; a query returns sensible results; indexing time reported honestly.

---

## Phase 4 — Retrieval pipeline + timing instrumentation

```
Follow the latency-instrumentation skill. Implement:

1. backend/app/benchmarking/latency.py — the stage_timer context manager and LatencyTrace model as described in the skill.
2. backend/app/retrieval/retriever.py — a Retriever class wrapping vector_store.query(), instrumented with stage_timer for "embed_query" and "vector_search" stages separately (embedding the query and searching the index are different costs, report them separately, don't merge them).
3. Wire a default chunking strategy as the "active" one for now (pick one, e.g. metadata_aware, note it's swappable via config) — full strategy comparison happens in Phase 8.
4. A quick manual timing script scripts/time_retrieval.py running 20 sample queries through the retriever and printing per-stage millisecond timings for each, plus a rough P50 across the 20 (full percentile reporting is Phase 8 — this is a sanity check).

Run scripts/time_retrieval.py and report actual millisecond numbers. If retrieval is not comfortably under 200ms at this stage, say so plainly rather than waiting until Phase 8 to discover it.
```

**Acceptance:** real timing numbers reported; retrieval demonstrably instrumented per-stage, not as one blob.

---

## Phase 5 — Speech-to-text integration

```
Implement in backend/app/stt/:

1. base.py — an STTProvider protocol: transcribe(audio_bytes: bytes, mime_type: str) -> TranscriptResult (text, confidence if available, language if available).
2. sarvam.py and elevenlabs.py — both implementing STTProvider, both real (not one stubbed), reading their API key from config. Use tenacity for retry-with-backoff on transient HTTP failures, and a request timeout (e.g. 10s) so a hung STT call can't hang the whole pipeline.
3. factory.py — get_stt_provider() reading STT_PROVIDER from config and returning the right instance.
4. A POST /transcribe endpoint in main.py (or a router) accepting an uploaded audio file, calling the configured provider, timing it with stage_timer("stt"), and returning the transcript.
5. Since we may not always have a live API key handy while developing, add a "mock" mode: if the configured provider's API key is empty in config, fall back to a MockSTTProvider that returns a fixed test transcript with a note in the response that it's mocked — so the endpoint never silently fails during dev, but also never pretends a mock result is real in a way that could leak into the final demo. Log a warning when mock mode is active.
6. Test hitting /transcribe with a small sample audio file (generate one with a TTS library or use a tiny fixture) in mock mode, asserting a 200 response and a transcript field.

Actually test this against whichever real STT key is present in .env, if one is — report whether the real call succeeded, and what the transcript looked like for a real short test clip.
```

**Acceptance:** `/transcribe` works in mock mode without keys; works against the real provider if a key is present; retries/timeout demonstrably present in code, not just described.

---

## Phase 6 — Harness / orchestrator + answer generation

```
Implement:

1. backend/app/generation/llm_client.py — wraps Groq (default) / Gemini (fallback, per LLM_PROVIDER config) behind a generate(prompt, context_chunks) -> GenerationResult function, with tenacity retry + timeout like the STT client. Groq is OpenAI-SDK compatible (base_url https://api.groq.com/openai/v1), so the `openai` python package works directly against it — just point it at Groq's base URL and key. Gemini needs the `google-genai` package instead; keep both behind the same interface so switching is a config change.
2. backend/app/generation/prompts.py — the prompt template instructing the model to answer ONLY from the provided context chunks and to say it doesn't know if the context doesn't contain the answer. Keep this genuinely strict — this is the first line of defense against ungrounded answers, before Phase 7's guardrails add a second, independent check.
3. backend/app/harness/orchestrator.py — the actual pipeline: accepts audio (or raw text, for testing without audio), runs STT (skip if text provided) -> retrieval -> generation, in sequence, with each stage timed via stage_timer, wrapped in a structured Pydantic response model (transcript, retrieved_chunks, answer, latency_trace, warnings: list[str]). Catch exceptions per stage and populate `warnings` instead of crashing the whole request when one stage fails gracefully-recoverably (e.g. STT fails -> can't proceed, return a clear error; a single retrieval match missing metadata -> log a warning, continue).
4. A POST /ask endpoint wiring the orchestrator end to end, accepting either an audio file or a raw text query (for easier testing/demo without a mic).
5. An integration test running a real (or mocked-STT) text query through the full orchestrator and asserting a well-formed response with a non-empty answer and a populated latency_trace.

Run a handful of real end-to-end queries (text input is fine) through /ask and report the actual answers you got, plus the latency_trace for each — don't describe hypothetical output.
```

**Acceptance:** `/ask` genuinely runs STT→retrieval→generation in sequence with structured output and real error handling; this is the first phase where the whole pipeline is demonstrably wired together, not simulated.

---

## Phase 7 — Guardrails

```
Follow the rag-guardrails skill. Implement:

1. backend/app/guardrails/input_filter.py, grounding_check.py, refusal.py as specified in the skill.
2. Wire input_filter into the orchestrator BEFORE the retrieval stage, and grounding_check AFTER generation but BEFORE returning the response — edit orchestrator.py directly, don't add a parallel guardrail path that the main flow bypasses.
3. When a guardrail trips, the /ask response should clearly indicate refusal (a `refused: bool` field plus a human-readable reason), not just return an empty or generic answer.
4. backend/tests/test_guardrails.py with the test set specified in the skill (off-topic, in-scope-should-NOT-refuse, unsafe input, engineered-ungrounded-answer cases) — at least 10 cases total across categories.

Run /guardrail-test-equivalent locally (pytest -k guardrail) and report the real pass/fail breakdown per category, including any false refusals on the in-scope test cases. If any category fails, fix it now — don't leave this for later since guardrails are an explicit grading criterion.
```

**Acceptance:** guardrail checks are demonstrably called inside `orchestrator.py`'s actual request path (verifiable via `/review-phase`); test suite passes including the false-refusal check.

---

## Phase 8 — Latency benchmarking (P50/P70/P100)

```
Implement scripts/run_benchmark.py:

1. Runs a configurable number of test queries (default 50, real questions relevant to the MSMARCO-XI dataset content — pull some from the actual dataset's questions if the schema has them, or write plausible ones based on the passages you've seen) through the orchestrator via the /ask endpoint or directly against orchestrator.py.
2. For each query, capture the full LatencyTrace.
3. Compute P50, P70, P100 for (a) retrieval-only (embed_query + vector_search + chunk_assembly stages) and (b) full end-to-end (all stages including stt and generation) — separately, per the latency-instrumentation skill.
4. Write results to docs/latency_report.md: a table of the percentiles, a short honest paragraph stating whether the retrieval-only number meets the sub-200ms target from the task spec, and a clearly separate honest statement of the full end-to-end numbers with no claim of meeting 200ms attached to them.
5. Also run the benchmark once per chunking strategy (all four from Phase 2) if time allows, and note in the report which strategy was fastest/slowest — this becomes evidence for the "vast chunking, not just one" requirement.

Actually run this against the real pipeline and report real numbers in docs/latency_report.md. Do not fabricate percentiles.
```

**Acceptance:** `docs/latency_report.md` exists with real numbers; retrieval-only vs. full end-to-end is honestly separated; `/latency-check` command works against this script.

---

## Phase 9 — Frontend + deployment

```
1. Minimal frontend/ (Vite + vanilla JS or lightweight React): a record button (MediaRecorder API) capturing audio, sending it to POST /ask, displaying the transcript, the answer, whether it was refused (and why, if so), and the latency breakdown for that request. Keep this deliberately simple — this is not being graded on visual polish.
2. CORS config in backend/app/main.py allowing the deployed frontend origin.
3. A Dockerfile (or Render/Railway-specific config, whichever platform was chosen in PREREQUISITES.md) for the backend, and deployment steps documented in README.md's Deployment section — actually deploy it and get a real live URL.
4. Update README.md with the real live link and confirm the /health endpoint responds at that URL.
5. Update docs/submission_checklist.md (create if it doesn't exist) with every item from the task's Submission Requirements section, checked off as completed.

Deploy for real and report back the actual live URL, and confirm you hit /health on it successfully (paste the real response).
```

**Acceptance:** a real, working public URL exists and responds; frontend can record voice and get a full answer back from the deployed backend; `README.md` and `docs/submission_checklist.md` reflect reality, not aspiration.

---

## After Phase 9

You still need, per the task doc — these are **not** OpenCode phases, they're human/team tasks:
- Fill the submission Google Form
- Record and post **Video 1** (90s process video) and **Video 2** (demo) — see `REVIEW.md`'s submission checklist for the exact posting requirements (Instagram + X, every member individually, `#RAGInGoa`, one public IG account minimum)
- Final read-through of `REVIEW.md`'s full checklist before submitting — **no resubmissions are allowed**, so this pass matters.