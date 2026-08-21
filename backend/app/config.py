from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root (backend/app/config.py -> backend -> repo root), so the same
# .env is found whether uvicorn/pytest is launched from backend/ or the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Speech-to-text ---
    stt_provider: str = "sarvam"
    sarvam_api_key: str = ""
    elevenlabs_api_key: str = ""

    # --- LLM generation ---
    llm_provider: str = "groq"
    groq_api_key: str = ""
    gemini_api_key: str = ""
    # Aug 2026: llama-3.3-70b-versatile no longer exists on Groq's free tier;
    # gpt-oss-20b is the current fast general-instruction model
    groq_model: str = "openai/gpt-oss-20b"

    # --- Vector store ---
    vector_store_path: str = str(_REPO_ROOT / "backend" / "data" / "index")

    @field_validator("vector_store_path", mode="after")
    @classmethod
    def _anchor_relative_paths(cls, v: str) -> str:
        # .env carries './backend/data/index' (relative). Resolving it against
        # the CWD made the store location depend on where uvicorn/pytest was
        # launched: from backend/ it silently pointed at a nonexistent
        # backend/backend/data/index, chroma auto-created an EMPTY store, and
        # every query returned zero context. Anchor relative paths to the repo
        # root so cwd can never move the index out from under the app.
        p = Path(v).expanduser()
        if not p.is_absolute():
            p = (_REPO_ROOT / p).resolve()
        return str(p)

    # --- Dataset ingestion ---
    dataset_name: str = "ai4bharat/MSMARCO-XI"
    dataset_language: str = "hin"
    dataset_split: str = "validation"
    max_raw_rows: int = 1000  # ~10 passages/row in MSMARCO-XI, so ~10k processed passages
    min_passage_chars: int = 30

    # --- Retrieval ---
    # Active chunking strategy for query-time retrieval. Swappable via config;
    # per-strategy comparison happens in the phase 8 benchmark.
    default_chunk_strategy: str = "metadata_aware"

    # --- Frontend serving ---
    # Comma-separated origins allowed cross-origin. Production is same-origin
    # (FastAPI serves frontend/dist), so this only matters for the Vite dev
    # server and any separately-hosted frontend.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    raw_data_dir: Path = _REPO_ROOT / "backend" / "data" / "raw"
    processed_data_dir: Path = _REPO_ROOT / "backend" / "data" / "processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()
