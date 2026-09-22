"""Central configuration. Same retrieval tunables as the hybrid-rag-api
project (this service reuses that retrieval core), plus agent-specific
settings for the ReAct loop's step budget."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "sample_docs"
INDEX_DIR = DATA_DIR / "index"


@dataclass(frozen=True)
class Settings:
    # Chunking
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "220"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "40"))

    # Embedding / reranker models
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    reranker_model_name: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # Retrieval
    bm25_top_k: int = int(os.getenv("BM25_TOP_K", "20"))
    dense_top_k: int = int(os.getenv("DENSE_TOP_K", "20"))
    fusion_top_k: int = int(os.getenv("FUSION_TOP_K", "10"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    # LLM (agent planner). Falls back to the heuristic agent when unset.
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Agent loop
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "5"))

    index_dir: Path = INDEX_DIR
    docs_dir: Path = DOCS_DIR


settings = Settings()
