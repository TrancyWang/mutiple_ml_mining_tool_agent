from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from academic_agent.infrastructure.runtime_paths import app_data_root


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class RAGSettings:
    data_dir: Path = Path(os.getenv("RAG_DATA_DIR", str(app_data_root() / "rag_datasets")))
    max_file_size: int = _int("RAG_MAX_FILE_SIZE", 50 * 1024 * 1024)
    embedding_dim: int = _int("RAG_EMBEDDING_DIM", 768)
    openai_base_url: str = os.getenv("RAG_OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", "")).rstrip("/")
    openai_api_key: str = os.getenv("RAG_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    chat_model: str = os.getenv("RAG_CHAT_MODEL", os.getenv("CHAT_MODEL", ""))
    embedding_base_url: str = os.getenv("RAG_EMBEDDING_BASE_URL", os.getenv("EMBEDDING_BASE_URL", "")).rstrip("/")
    embedding_api_key: str = os.getenv("RAG_EMBEDDING_API_KEY", os.getenv("EMBEDDING_API_KEY", ""))
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", ""))
    timeout: int = _int("RAG_TIMEOUT", 90)

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir


settings = RAGSettings()
