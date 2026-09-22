"""Sparse (BM25) retrieval — good at exact keyword / entity matches that
dense embeddings can sometimes blur (e.g. "5,000 events per second",
specific header names, model version numbers)."""
from __future__ import annotations

from app.models import Chunk
from app.store import RagIndex
from app.text_utils import tokenize_for_bm25


def bm25_search(index: RagIndex, query: str, top_k: int) -> list[tuple[Chunk, float]]:
    tokenized_query = tokenize_for_bm25(query)
    scores = index.bm25.get_scores(tokenized_query)

    ranked_indices = scores.argsort()[::-1][:top_k]
    results = [(index.chunks[i], float(scores[i])) for i in ranked_indices if scores[i] > 0]
    return results
