"""Index build/persist/load.

Builds two parallel indexes over the same chunk set:
  - a BM25 sparse index (rank_bm25) for exact keyword matching
  - a dense embedding index (sentence-transformers, cosine similarity via
    normalized dot product) for semantic matching

Both are persisted to disk under data/index/ so the API can start up by
loading a prebuilt index rather than re-embedding on every boot.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import Settings, settings
from app.ingestion import build_chunks
from app.models import Chunk
from app.text_utils import normalize_vectors, tokenize_for_bm25

CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"
BM25_FILE = "bm25.pkl"


class RagIndex:
    """In-memory handle to the built chunks, BM25 index, and dense embeddings."""

    def __init__(
        self,
        chunks: list[Chunk],
        bm25: BM25Okapi,
        embeddings: np.ndarray,
        embedder: SentenceTransformer,
    ) -> None:
        self.chunks = chunks
        self.bm25 = bm25
        self.embeddings = embeddings  # shape (n_chunks, dim), L2-normalized
        self.embedder = embedder

    @property
    def size(self) -> int:
        return len(self.chunks)


def build_index(cfg: Settings | None = None) -> RagIndex:
    cfg = cfg or settings
    chunks = build_chunks(cfg.docs_dir, cfg)

    tokenized_corpus = [tokenize_for_bm25(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    embedder = SentenceTransformer(cfg.embedding_model_name)
    raw_embeddings = embedder.encode(
        [c.text for c in chunks],
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    embeddings = normalize_vectors(raw_embeddings)

    return RagIndex(chunks=chunks, bm25=bm25, embeddings=embeddings, embedder=embedder)


def save_index(index: RagIndex, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    chunks_payload = [c.model_dump() for c in index.chunks]
    (index_dir / CHUNKS_FILE).write_text(json.dumps(chunks_payload, indent=2))

    np.save(index_dir / EMBEDDINGS_FILE, index.embeddings)

    with open(index_dir / BM25_FILE, "wb") as f:
        pickle.dump(index.bm25, f)


def load_index(cfg: Settings | None = None) -> RagIndex:
    cfg = cfg or settings
    index_dir = cfg.index_dir

    chunks_path = index_dir / CHUNKS_FILE
    embeddings_path = index_dir / EMBEDDINGS_FILE
    bm25_path = index_dir / BM25_FILE

    if not (chunks_path.exists() and embeddings_path.exists() and bm25_path.exists()):
        raise FileNotFoundError(
            f"No prebuilt index found in {index_dir}. Run `python scripts/ingest.py` first."
        )

    chunks = [Chunk(**c) for c in json.loads(chunks_path.read_text())]
    embeddings = np.load(embeddings_path)
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    embedder = SentenceTransformer(cfg.embedding_model_name)

    return RagIndex(chunks=chunks, bm25=bm25, embeddings=embeddings, embedder=embedder)
