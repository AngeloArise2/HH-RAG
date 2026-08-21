# --- stage 1: build the frontend -------------------------------------------
FROM node:20-alpine AS frontend-build
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- stage 2: backend runtime -----------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    OMP_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only torch FIRST: unpinned sentence-transformers would otherwise pull
# ~2GB of CUDA wheels into a 512MB-RAM container image. This pin keeps the
# runtime wheel at ~200MB and is the only torch in the image (requirements.txt
# sees torch already satisfied).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake MiniLM weights into the image so container startup touches no network.
# Must use the same model id as app/retrieval/embed.py:EMBEDDING_MODEL.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# App + built frontend + pre-built retrieval index (extracted at BUILD time:
# boot does zero extraction, zero download — just warmup, then ready)
COPY backend/app ./backend/app
COPY --from=frontend-build /fe/dist ./frontend/dist
COPY backend/data/index_snapshot.tgz ./backend/data/index_snapshot.tgz
RUN mkdir -p backend/data && tar -xzf backend/data/index_snapshot.tgz -C backend/data \
    && rm backend/data/index_snapshot.tgz

# config.py anchors relative paths to the repo root, so the snapshot lands
# exactly where settings.vector_store_path points regardless of cwd.
ENV PYTHONPATH=/app/backend

# Hugging Face Spaces routes to app_port declared in README front matter
# (7860 is the Spaces default); PORT env override kept for local runs.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
