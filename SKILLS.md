# SKILLS.md — Concepts, in plain English

This is **not** an OpenCode config file — it's for you, the human. A glossary of every concept this project touches, written so you can follow along without a CS background. Keep this open in a tab. When you run `/explain-last` (see `REVIEW.md`), the agent will lean on this same vocabulary.

---

### RAG (Retrieval-Augmented Generation)
Instead of asking an AI a question and hoping it "remembers" the right facts, you first **fetch** the most relevant snippets of text from your own dataset, then hand those snippets to the AI along with the question, and say "answer using *this*." It's open-book exam vs. closed-book exam. The "retrieval" part is the search; the "generation" part is the AI writing the answer.

### Speech-to-text (STT)
Converts a recorded voice clip into text. Sarvam and ElevenLabs are two companies whose APIs do this — you send audio, you get back a transcript.

### Chunking
Your dataset is huge — you can't hand the whole thing to the AI every time. So you cut it into small pieces ("chunks") ahead of time and store them. Chunking strategy = *how* you cut:
- **Fixed-size:** every chunk is, say, exactly 200 words, no matter where sentences fall. Simple, but can slice a sentence in half.
- **Overlap:** neighboring chunks share a few sentences, so an idea that spans a chunk boundary isn't lost.
- **Semantic / sentence-aware:** cut at natural sentence or topic boundaries instead of a fixed word count, so chunks stay meaningful.
- **Metadata-aware:** keep track of *where* a chunk came from (document title, section, position) so you can filter or boost results later, not just treat every chunk as anonymous text.

The task explicitly wants more than one of these — a "vast" strategy, in their words — because a single naive approach is the easy/lazy answer.

### Embeddings
A way of turning a chunk of text (or a question) into a list of numbers (a "vector") that captures its *meaning*, not its exact wording. Two sentences that mean similar things end up with similar number-lists, even if they don't share a single word.

### Vector database (vector DB)
A specialized storage system for those number-lists, built to answer "which of these thousands of stored vectors is closest in meaning to *this* new vector?" — fast. That's how retrieval works: turn the user's question into a vector, ask the vector DB for the closest chunks, hand those chunks to the AI.

### Retrieval
The step of actually querying the vector DB and getting back the top-matching chunks for a given question.

### Latency / P50 / P70 / P100
Latency = how long something takes, usually in milliseconds (ms). Instead of reporting one lucky fast run, you run the pipeline many times and report **percentiles**:
- **P50** (median) — half your queries were faster than this, half slower. Your "typical" speed.
- **P70** — 70% of queries were this fast or faster.
- **P100** — the slowest single query you saw (worst case).
Reporting all three (not just your best run) shows the judges your real-world performance, warts included.

### Harness
The scaffolding *around* the AI model — the code that calls STT, then retrieval, then the AI, in order; handles it if an API call fails and retries; makes sure the data passed between steps is in the right shape; and doesn't just crash or hang if something goes wrong. A "raw prompt-in, text-out call" (what the task says *not* to submit) is just calling the AI directly with no scaffolding — fragile and unstructured.

### Guardrails
Rules and checks that stop the system from doing the wrong thing:
- **Off-topic detection** — if someone asks something unrelated to the dataset, the system says "I can't help with that" instead of making something up.
- **Unsafe-input handling** — catching inappropriate or malicious input before it reaches the AI.
- **Hallucination / groundedness check** — after the AI generates an answer, checking whether that answer is actually supported by the retrieved chunks, and refusing or flagging it if it isn't. This is what "knows when not to answer" means in the task doc.

### Orchestration
Just a fancier word for "the harness manages the order of operations and the handoffs between steps."

### Deployment / "live working link"
Taking your code, which currently only runs on your laptop, and putting it on a server somewhere on the internet so anyone with the link can use it. We're using Render/Railway/Fly (backend) and optionally Vercel (frontend) because they're free-tier friendly and fast to set up.

### Why 200ms is aggressive
For reference: a single round-trip network request to a cloud API often takes 100–300ms *by itself*, and generating a full sentence of text with an LLM usually takes 1–5+ seconds. Chunking and a local vector DB search, by contrast, genuinely can run in single-digit-to-low-double-digit milliseconds. That's why `AGENTS.md` and `README.md` are explicit that the 200ms number is being honestly measured against the **retrieval-side** pipeline, with full end-to-end reported separately — not quietly shrunk to fit.
