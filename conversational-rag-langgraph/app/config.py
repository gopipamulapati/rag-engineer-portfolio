"""Central configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "sample_docs"


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = Path(os.getenv("DOCS_DIR", str(DOCS_DIR)))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "3"))

    # Memory: "memory" (in-process) or "sqlite" (survives restarts)
    memory_backend: str = os.getenv("MEMORY_BACKEND", "memory")
    sqlite_path: str = os.getenv("SQLITE_PATH", "sessions.db")
    # How many previous turns the condenser looks at
    history_turns: int = int(os.getenv("HISTORY_TURNS", "3"))

    # Self-check: share of answer content words that must appear in the
    # retrieved context for the answer to count as grounded
    grounding_threshold: float = float(os.getenv("GROUNDING_THRESHOLD", "0.6"))
    max_retrieval_attempts: int = int(os.getenv("MAX_RETRIEVAL_ATTEMPTS", "2"))

    # Optional LLM. With no key, the graph uses deterministic heuristics.
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


settings = Settings()
