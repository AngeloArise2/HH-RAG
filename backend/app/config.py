from functools import lru_cache
from pathlib import Path

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
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Vector store ---
    vector_store_path: str = str(_REPO_ROOT / "backend" / "data" / "index")

    # --- Dataset ingestion ---
    dataset_name: str = "ai4bharat/MSMARCO-XI"
    dataset_language: str = "hin"
    dataset_split: str = "validation"
    max_raw_rows: int = 10000
    min_passage_chars: int = 30

    raw_data_dir: Path = _REPO_ROOT / "backend" / "data" / "raw"
    processed_data_dir: Path = _REPO_ROOT / "backend" / "data" / "processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()
