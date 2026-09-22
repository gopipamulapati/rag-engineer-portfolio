"""Cross-encoder reranking.

Bi-encoders (used for the dense retrieval step) embed the query and each
chunk independently, which is fast enough to search thousands of chunks but
loses some precision because the query and chunk never directly attend to
each other. A cross-encoder takes the (query, chunk) pair together and
scores relevance directly, which is far more accurate but too slow to run
over the whole corpus -- so it's applied only to the fusion_top_k
candidates that already survived hybrid retrieval, as a final precision
pass before returning results to the caller.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import Settings, settings
from app.models import ScoredChunk


@lru_cache(maxsize=1)
def _get_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank(
    query: str, candidates: list[ScoredChunk], cfg: Settings | None = None
) -> list[ScoredChunk]:
    cfg = cfg or settings
    if not candidates:
        return candidates

    reranker = _get_reranker(cfg.reranker_model_name)
    pairs = [(query, c.chunk.text) for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate.rerank_score = float(score)

    reranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
    return reranked[: cfg.rerank_top_k]
