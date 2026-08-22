# Submission checklist

Mirrors `REVIEW.md`'s final checklist (task doc requirements) with the actual
completion state. Items marked ⏳ are human tasks — code/deploy work alone
can't complete them. **No resubmissions allowed — one-shot check before Aug 22, 2026, 11:59 PM.**

## Deployed system

- [x] Docker image builds from repo root `Dockerfile` (multi-stage: node frontend build → python-slim backend)
- [x] Image verified locally: `/health` responds, real grounded answer via `/ask`, SPA served at `/`
- [x] Retrieval index ships inside the image (`backend/data/index_snapshot.tgz`, ~33MB, extracted at build time) — deploys need no dataset download or re-embedding
- [x] Frontend served same-origin by the backend (record button → MediaRecorder webm → POST `/ask`; transcript / answer / refusal+reason / grounding-verified caveat / latency table rendered)
- [x] CORS configured for dev (`CORS_ORIGINS` env; prod is same-origin so no cross-origin requests exist)
- [x] Platform chosen on measurement: Railway (trial credit) — Render free's 512MB cap measured insufficient (~483MB anonymous idle memory in the built image), HF Spaces Docker PRO-paywalled July 2026; see README Deployment section
- [x] Railway service deployed, public domain live: https://hh-rag-production.up.railway.app — secrets set (GROQ_API_KEY, SARVAM_API_KEY)
- [x] Live URL verified (Aug 22, 2026): `/health` → `{"status":"ok"}`; real grounded `/ask` answer (retrieval 23.5ms); off-topic refusal short-circuits at `guardrail_check`
- [x] README "Deployment" section filled with the REAL live URL + real health response
- [ ] ⏳ Voice query end-to-end through the deployed UI in a browser (mic permission) — *do once right before recording the demo video*

## Task-doc requirements (from REVIEW.md final checklist)

- [ ] ⏳ Submission form filled: https://forms.gle/MNvCjcv23Hn2Eeu58
- [ ] GitHub repo link accessible to reviewers *(repo is public at github.com/AngeloArise2/HH-RAG — verify before submit)*
- [ ] Live link tested fresh at submission time (Space may have slept — hit `/health` first)
- [ ] ⏳ Video 1 (team/process, 90s) recorded and posted by every member to Instagram + X with #RAGInGoa
- [ ] ⏳ Video 2 (demo) recorded and posted by every member to Instagram + X with #RAGInGoa — demo must show at least one guardrail refusal, not just happy paths
- [ ] ⏳ At least one team member's Instagram account public
- [x] STT provider unambiguous: **Sarvam** (`STT_PROVIDER=sarvam`; ElevenLabs exists behind the interface but is not active)
- [x] Multi-strategy chunking genuine: 4 strategies (fixed-size, fixed-size+overlap, semantic, metadata-aware) behind one `Chunker` interface, per-strategy tests, preview script, strategy comparison in `docs/latency_report.md`
- [x] P50/P70/P100 latency numbers from a real multi-query run in `docs/latency_report.md` (retrieval-only P50 31.2 / P70 32.5 / P100 49.3ms vs 200ms target = MEETS; e2e reported separately with explicit no-claim header)
- [x] Harness/orchestration visible: `backend/app/harness/orchestrator.py` runs STT → input-guardrail → retrieval → generation → grounding-check with per-stage timing, retries/timeouts on external calls, structured Pydantic responses
- [x] Guardrails demonstrable: off-topic/unsafe/ungrounded refusals wired into the request path (12/40 benchmark queries refused honestly); false-refusal checks in test suite

## Pre-submit verification commands

```bash
.venv/bin/python -m pytest backend/tests -q        # expect: all passing
curl https://<railway-domain>.up.railway.app/health  # expect: {"status":"ok"}
```
