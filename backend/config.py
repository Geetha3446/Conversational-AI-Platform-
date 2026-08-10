"""
Central configuration for the whole backend.

Everything the app needs to know about its environment lives here, loaded once
from the .env file at the project root. Import `settings` anywhere you need a
value; never read os.environ directly elsewhere, so there is a single place to
look when something is misconfigured.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the folder that contains "backend/", "frontend/", ".env"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed view of the .env file. Pydantic validates and casts for us."""

    # --- LLM ---------------------------------------------------------------
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.3

    # --- Auth --------------------------------------------------------------
    JWT_SECRET_KEY: str = "insecure-dev-key-change-me"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 days

    # --- Storage -----------------------------------------------------------
    DATA_DIR: str = "data"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Networking --------------------------------------------------------
    BACKEND_URL: str = "http://127.0.0.1:8000"

    # --- RAG tuning --------------------------------------------------------
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    RETRIEVER_K: int = 4

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # tolerate unknown keys in .env instead of crashing
    )

    # --- Derived paths -----------------------------------------------------
    @property
    def data_path(self) -> Path:
        """Absolute path to the data directory, created on first access."""
        p = PROJECT_ROOT / self.DATA_DIR
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        """Single SQLite file holding users, chats, messages AND checkpoints."""
        return self.data_path / "app.db"

    @property
    def sqlalchemy_url(self) -> str:
        return f"sqlite:///{self.db_path}"

    @property
    def uploads_path(self) -> Path:
        p = self.data_path / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def vectorstore_path(self) -> Path:
        p = self.data_path / "vectorstores"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def user_upload_dir(self, user_id: int) -> Path:
        """Every user gets a private folder for their PDFs."""
        p = self.uploads_path / f"user_{user_id}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def user_index_dir(self, user_id: int) -> Path:
        """Every user gets a private FAISS index, so RAG never leaks across users."""
        p = self.vectorstore_path / f"user_{user_id}"
        p.mkdir(parents=True, exist_ok=True)
        return p


# A single shared instance, imported everywhere.
settings = Settings()
