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
- [ ] `stage_timer` / `LatencyTrace` implemented and used (not scattered `time.time()` calls)
- [ ] `embed_query` and `vector_search` timed **separately**
- [ ] Real millisecond numbers reported from a real run
- [ ] Honest statement if retrieval isn't yet comfortably under 200ms

### Phase 5 — STT
- [ ] Both Sarvam and ElevenLabs implemented behind one interface (even though only one is "active")
- [ ] Retry-with-backoff and timeout present in code
- [ ] Mock fallback works with no key, clearly flagged as mock (not silently passed off as real)
- [ ] Real provider tested if a key is present, real transcript reported

### Phase 6 — Harness / orchestrator + generation
- [ ] `/ask` runs STT → retrieval → generation in real sequence
- [ ] Structured (Pydantic) response, not a loose dict
- [ ] Per-stage error handling — one stage failing doesn't silently corrupt the whole response
- [ ] Real end-to-end queries run and real answers/latency reported

### Phase 7 — Guardrails
- [ ] `input_filter` called before retrieval **inside the orchestrator**, verified in code, not just present as an unused function
- [ ] `grounding_check` called after generation, before the response is returned
- [ ] `refused` field + human-readable reason on refusal
- [ ] Guardrail test suite passes: off-topic, unsafe input, ungrounded-answer, AND in-scope-should-not-refuse cases
- [ ] No false refusals on the in-scope test cases

### Phase 8 — Latency benchmarking
- [ ] `docs/latency_report.md` has real P50/P70/P100 from a real run (≥30-50 queries)
- [ ] Retrieval-only and full end-to-end reported **separately**, both honestly
- [ ] Retrieval-only number checked against the 200ms target explicitly
- [ ] No target claimed for full end-to-end that isn't actually met

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
