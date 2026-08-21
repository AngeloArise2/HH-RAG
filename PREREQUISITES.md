# PREREQUISITES — Day 0 Setup

Do this **before** you open `BUILD_PROMPT.md`. Budget ~60–90 minutes. Deadline is Aug 22, 11:59 PM — you have roughly one day, so Day 0 setup should happen tonight, not tomorrow morning.

---

## 1. Accounts & API keys (get these first — signup approval can lag)

| Service | Why you need it | Get it here |
|---|---|---|
| **Sarvam AI** *or* **ElevenLabs** | Speech-to-text (pick ONE, per task spec) | sarvam.ai (API key from dashboard) or elevenlabs.io |
| **Anthropic or OpenAI** | The LLM that generates the final answer | console.anthropic.com or platform.openai.com |
| **GitHub** | Repo hosting (mandatory submission item) | You likely have this |
| **Hugging Face** | Dataset access (`ai4bharat/MSMARCO-XI`) | huggingface.co — account not always required for public datasets, but create one anyway in case it's gated |
| **Render, Railway, or Fly.io** | Backend deployment (need a "live working link") | Pick one now, don't waffle later |
| **Vercel or Netlify** | Frontend deployment (optional if serving frontend from same backend) | Only if you split frontend/backend |
| **Instagram** (personal, one must be public) | Mandatory promo posts | — |
| **X (Twitter)** | Mandatory promo posts | — |

**Decision to make right now:** Sarvam vs ElevenLabs.
- **Sarvam** — built for Indian languages, likely cheaper/free-tier friendlier, good if your test queries are Hindi/Indic-language influenced (dataset is MS MARCO-**XI**, i.e. Indic).
- **ElevenLabs** — excellent English STT, more familiar docs, but not Indic-optimized.
Given the dataset name (MSMARCO-**XI** = Indic), **Sarvam is the safer default** unless your team already has ElevenLabs credits. Pick one and write it into `PREREQUISITES.md` / `README.md` once decided — don't leave both wired in, it wastes review time.

---

## 2. Local software (install tonight)

- **Node.js** ≥ 20 (`node -v`)
- **Python** ≥ 3.10 (`python3 --version`)
- **Git**
- **ffmpeg** (audio format conversion — `ffmpeg -version`; install via `brew install ffmpeg` / `apt install ffmpeg`)
- **OpenCode** itself: `curl -fsSL https://opencode.ai/install | bash` (or see opencode.ai/docs — you're using it in VS Code, so also install the OpenCode VS Code extension if you want inline diffs)
- A vector DB library — **no server to install**, we're using an in-process store (Chroma or FAISS) specifically because it's the fastest option for the 200ms retrieval target. Don't install a hosted vector DB (Pinecone, Weaviate Cloud) — the network hop alone risks blowing your latency budget.

---

## 3. Repo setup

```bash
mkdir hh-goa-voice-rag && cd hh-goa-voice-rag
git init
gh repo create hh-goa-voice-rag --private --source=. --remote=origin   # or create on github.com manually
```

Drop in the seven root files from this kit (`AGENTS.md`, `README.md`, `PREREQUISITES.md`, `BUILD_PROMPT.md`, `REVIEW.md`, `SKILLS.md`, `OPENCODE_SKILLS.md`) plus the `.opencode/` folder, then:

```bash
git add .
git commit -m "chore: project scaffolding + opencode workflow"
git push -u origin main
```

Open the folder in VS Code, open the OpenCode panel/terminal inside it.

---

## 4. `.env` — create this now, fill keys as you get them

```bash
cat > .env.example << 'EOF'
# --- Speech-to-text (pick ONE provider, set the other's key blank) ---
STT_PROVIDER=sarvam            # sarvam | elevenlabs
SARVAM_API_KEY=
ELEVENLABS_API_KEY=

# --- LLM for answer generation ---
LLM_PROVIDER=anthropic         # anthropic | openai
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# --- Vector store (local, no key needed for Chroma) ---
VECTOR_STORE=chroma
VECTOR_STORE_PATH=./backend/data/index

# --- App ---
APP_ENV=development
LOG_LEVEL=info
EOF
cp .env.example .env
```

Fill `.env` with real keys. **Never commit `.env`** — confirm `.gitignore` has it (the scaffold's does).

---

## 5. Team logistics (do this now, not at 11 PM tomorrow)

- Assign roles: someone owns STT+guardrails, someone owns chunking+retrieval, someone owns harness+generation, someone owns frontend+deploy+videos. Phases in `BUILD_PROMPT.md` can run in parallel branches after Phase 0/1.
- Decide **who records Video 1 (90s process video) early** — capture footage *while you work*, don't reconstruct it at the end.
- Every team member needs to know: both videos, on **both Instagram and X**, individually, every post tagged **#RAGInGoa**, at least one Instagram account must be public. Put a name next to each of these in a shared doc now.

---

## 6. Sanity check before starting Phase 0

- [ ] `opencode` runs in your project folder
- [ ] `.env` has at least one STT key and one LLM key filled in
- [ ] You can `curl` huggingface.co (dataset access isn't blocked on your network)
- [ ] GitHub repo exists and you can push
- [ ] Deployment account (Render/Railway/Fly) created, no need to deploy yet

Once all boxes are checked, open `BUILD_PROMPT.md` and paste **Phase 0** into OpenCode.
