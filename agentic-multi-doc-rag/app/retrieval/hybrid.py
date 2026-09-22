"""Fuse BM25 and dense retrieval results using Reciprocal Rank Fusion (RRF).

RRF is used instead of a weighted sum of raw scores because BM25 scores and
cosine similarities live on completely different, unnormalized scales --
naively averaging them lets whichever retriever happens to produce larger
numbers dominate. RRF instead only looks at *rank position* within each
list, which makes it scale-free and robust without needing to hand-tune a
score-blending weight per dataset.

score(chunk) = sum over retrievers r that returned this chunk of 1 / (k + rank_r(chunk))
"""
from __future__ import annotations

from app.config import Settings, settings
from app.models import Chunk, ScoredChunk
from app.retrieval.bm25_retriever import bm25_search
from app.retrieval.dense_retriever import dense_search
from app.store import RagIndex


def hybrid_search(index: RagIndex, query: str, cfg: Settings | None = None) -> list[ScoredChunk]:
    cfg = cfg or settings

    bm25_results = bm25_search(index, query, cfg.bm25_top_k)
    dense_results = dense_search(index, query, cfg.dense_top_k)

    bm25_rank = {chunk.chunk_id: rank for rank, (chunk, _) in enumerate(bm25_results)}
    bm25_score_by_id = {chunk.chunk_id: score for chunk, score in bm25_results}

    dense_rank = {chunk.chunk_id: rank for rank, (chunk, _) in enumerate(dense_results)}
    dense_score_by_id = {chunk.chunk_id: score for chunk, score in dense_results}

    chunk_by_id: dict[str, Chunk] = {}
    for chunk, _ in bm25_results:
        chunk_by_id[chunk.chunk_id] = chunk
    for chunk, _ in dense_results:
        chunk_by_id[chunk.chunk_id] = chunk

    fused_scores: dict[str, float] = {}
    for chunk_id in chunk_by_id:
        score = 0.0
        if chunk_id in bm25_rank:
            score += 1.0 / (cfg.rrf_k + bm25_rank[chunk_id] + 1)
        if chunk_id in dense_rank:
            score += 1.0 / (cfg.rrf_k + dense_rank[chunk_id] + 1)
        fused_scores[chunk_id] = score

    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    ranked_ids = ranked_ids[: cfg.fusion_top_k]

    scored_chunks = [
        ScoredChunk(
            chunk=chunk_by_id[cid],
            bm25_score=bm25_score_by_id.get(cid),
            dense_score=dense_score_by_id.get(cid),
            fused_score=fused_scores[cid],
        )
        for cid in ranked_ids
    ]
    return scored_chunks
