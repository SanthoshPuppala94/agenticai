from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("GRAPH_RAG_DATA_DIR", PROJECT_ROOT / "data"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    request_timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30"))


settings = Settings()
