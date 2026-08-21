# REVIEW.md

Two jobs: (1) a checklist to tick off as you complete each `BUILD_PROMPT.md` phase, and (2) the final submission checklist against the actual task requirements. Update section (1) as you go — don't leave it for the end.

## How to review a phase

1. Finish the phase's prompt in OpenCode.
2. Run `/review-phase` — read-only, flags gaps against the criteria below without touching code.
3. Address anything it flags.
4. Run `/explain-last` if you want the plain-English version of what happened (see `SKILLS.md` for vocabulary).
5. Check the boxes below.
6. Commit.

---

## Phase checklist

### Phase 0 — Scaffold
- [x] `uvicorn app.main:app` boots with no errors
- [x] `GET /health` returns 200
- [x] `pytest backend/tests` passes
- [x] Directory structure matches `AGENTS.md`
- [x] `.gitignore` excludes `.env`, `backend/data/*`, `__pycache__`

### Phase 1 — Dataset ingestion
- [x] Real subset of `ai4bharat/MSMARCO-XI` downloaded, not simulated
- [x] Real row count / stats reported (not guessed)
- [x] `processed/passages.jsonl` has unique `doc_id`s, no empty passages
- [x] Test passes

### Phase 2 — Multi-strategy chunking
- [x] 4 strategies exist: fixed-size, fixed-size+overlap, semantic, metadata-aware
- [x] All implement one common `Chunker` interface
- [x] Per-strategy tests pass (no empty chunks, max length respected, overlap correct)
- [x] Preview script shows real, visibly different output per strategy

### Phase 3 — Embedding + vector index
- [x] Local embedding model chosen and justified (no network hop in hot path)
- [x] 4 Chroma collections built (one per strategy) against real ingested data
- [x] Round-trip query test passes (relevant chunk retrieved for a relevant query)
- [x] Real indexing time reported

### Phase 4 — Retrieval + timing
- [x] `stage_timer` / `LatencyTrace` implemented and used (not scattered `time.time()` calls)
- [x] `embed_query` and `vector_search` timed **separately**
- [x] Real millisecond numbers reported from a real run
- [x] Honest statement if retrieval isn't yet comfortably under 200ms *(is comfortably under: P50 ≈ 22ms over 20 real queries — full P50/P70/P100 report still due in Phase 8)*

### Phase 5 — STT
- [x] Both Sarvam and ElevenLabs implemented behind one interface (even though only one is "active") *(Sarvam live; ElevenLabs real implementation, no key — mock-fallback tested)*
- [x] Retry-with-backoff and timeout present in code *(3 attempts, 0.5s→1s backoff, 10s timeout; 429/5xx/timeouts retried, 401/413 fail fast — all covered by httpx.MockTransport tests)*
- [x] Mock fallback works with no key, clearly flagged as mock (not silently passed off as real) *(is_mock flag + mock_note on /transcribe + warning log)*
- [x] Real provider tested if a key is present, real transcript reported *(Sarvam saaras:v3 on backend/data/audio/sample.webm: "Hello. So I just wanted to talk about how's the weather going today." en-IN, ~1.29s incl. network)*

### Phase 6 — Harness / orchestrator + generation
- [x] `/ask` runs STT → retrieval → generation in real sequence *(text input skips STT by design; audio path tested via MockSTTProvider)*
- [x] Structured (Pydantic) response, not a loose dict *(AskResponse: transcript/answer/chunks/latency_trace/warnings/is_mock flags)*
- [x] Per-stage error handling — one stage failing doesn't silently corrupt the whole response *(STT/retrieval failure → structured 502; LLM failure → chunks + empty answer + warning; missing metadata → warning; all covered in tests)*
- [x] Real end-to-end queries run and real answers/latency reported *(Groq openai/gpt-oss-20b, live: incorporation + taxes answered from context; "harry potter" correctly refused with "I don't know based on the provided context"; steady-state retrieval ≈ 24-28ms, generation ≈ 0.8-1.7s, reported separately)*

### Phase 7 — Guardrails
- [x] `input_filter` called before retrieval **inside the orchestrator**, verified in code, not just present as an unused function *(run_pipeline stage 2; live proof: refused queries show only `guardrail_check` in the trace — embed/vector_search never ran)*
- [x] `grounding_check` called after generation, before the response is returned *(stage 5; unsupported verdict swaps answer for refusal before AskResponse assembly)*
- [x] `refused` field + human-readable reason on refusal *(AskResponse.refused + refusal_reason + plain-language refusal strings per trip type)*
- [x] Guardrail test suite passes: off-topic, unsafe input, ungrounded-answer, AND in-scope-should-not-refuse cases *(22 tests in test_guardrails.py: 4 off-topic, 3 in-scope false-refusal checks, 2 unsafe, 2 engineered-ungrounded, plus parser/wiring/fail-open units)*
- [x] No false refusals on the in-scope test cases *(0/3 live and offline; live run also verified real LLM guards: gpt-oss needed reasoning_effort=low + larger token budget or guard calls came back empty)*

### Phase 8 — Latency benchmarking
- [x] `docs/latency_report.md` has real P50/P70/P100 from a real run (≥30-50 queries) *(40 real `Eng_Query` dataset questions + 5 voice runs through the live orchestrator; raw rows kept in backend/data/benchmark_results.json)*
- [x] Retrieval-only and full end-to-end reported **separately**, both honestly *(retrieval-only P50 31.2 / P70 32.5 / P100 49.3ms; text e2e P50 1713ms / P100 2148ms — no target attached to e2e anywhere; explicit Requirement-3 interpretation section explains why mandated external STT makes literal full-pipeline <200ms unsatisfiable)*
- [x] Retrieval-only number checked against the 200ms target explicitly *(verdict line: MEETS target, ~4× headroom even at P100)*
- [x] No target claimed for full end-to-end that isn't actually met *(e2e tables carry an explicit "NO 200ms claim" header; patterns-considered section documents adopted vs rejected optimizations with corrected math)*

### Phase 9 — Frontend + deployment
- [ ] Voice recording UI works and calls `/ask`
- [ ] Real live URL deployed and responding at `/health`
- [ ] `README.md` updated with the real live link
- [ ] `docs/submission_checklist.md` reflects actual completed state

---

## OpenCode commands (see `OPENCODE_SKILLS.md` for full detail)

| Command | Use it |
|---|---|
| `/review-phase` | After every phase, before committing |
| `/explain-last` | Whenever you (the human) don't understand what just got built |
| `/latency-check` | After Phase 8, and again before final submission |
| `/guardrail-test` | After Phase 7, and again before final submission |

---

## Final submission checklist (from the task doc — do not skip any line)

- [ ] **Submission form filled:** https://forms.gle/MNvCjcv23Hn2Eeu58
- [ ] **GitHub repo link** included, repo is accessible to reviewers (public, or shared with correct access)
- [ ] **Live working link** included and actually responds when tested fresh (not just "worked when I last checked")
- [ ] **Video 1 (Team/process, 90 seconds)** — shows the *process*, not the finished product
- [ ] **Video 2 (Demo)** — full end-to-end working demo
- [ ] Video 1 posted to **Instagram** by **every individual team member** (not one shared team post)
- [ ] Video 1 posted to **X** by **every individual team member**
- [ ] Video 2 posted to **Instagram** by **every individual team member**
- [ ] Video 2 posted to **X** by **every individual team member**
- [ ] Every single one of those posts includes **#RAGInGoa**
- [ ] **At least one team member's Instagram account is public**
- [ ] STT provider used is clearly **either** Sarvam **or** ElevenLabs (not ambiguous/mixed in the writeup)
- [ ] Chunking writeup/README section shows genuine multiple-strategy thought, not one fixed-size approach dressed up
- [ ] P50/P70/P100 latency numbers are in the submission, from a real multi-query run
- [ ] Harness/orchestration is visible in the repo (not a single raw prompt call)
- [ ] Guardrail behavior is demonstrable in the demo video (show at least one refusal, not just successful answers)
- [ ] Submitted **before Aug 22, 2026, 11:59 PM** — remember, **no resubmissions allowed**, so this is a one-shot check
