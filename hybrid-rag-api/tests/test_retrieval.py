"""Unit tests for the retrieval pipeline.

Uses the real sample_docs corpus (small enough to embed quickly) rather than
mocks, so these tests double as a smoke test that the whole pipeline -
chunking, BM25, embeddings, fusion, and reranking - actually works together,
not just in isolation.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import chunk_text, split_into_sentences
from app.retrieval.bm25_retriever import bm25_search
from app.retrieval.dense_retriever import dense_search
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank
from app.store import build_index


@pytest.fixture(scope="module")
def index():
    return build_index()


def test_split_into_sentences_basic():
    text = "First sentence. Second sentence! Third one?"
    sentences = split_into_sentences(text)
    assert sentences == ["First sentence.", "Second sentence!", "Third one?"]


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"word{i}." for i in range(100))
    chunks = chunk_text(text, chunk_size_tokens=20, chunk_overlap_tokens=5)

    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert len(chunk.split()) <= 20 + 5  # allow one sentence's slack


def test_bm25_finds_exact_keyword_match(index):
    results = bm25_search(index, "5,000 events per second", top_k=5)
    assert len(results) > 0
    top_chunk, _ = results[0]
    assert "5,000" in top_chunk.text or "events per second" in top_chunk.text


def test_dense_search_finds_semantic_match(index):
    # No shared keywords with the source text, but semantically about failover.
    results = dense_search(index, "how do we recover if a whole region goes down?", top_k=5)
    sources = [c.source for c, _ in results]
    assert "incident_response_runbook.txt" in sources


def test_hybrid_search_returns_scored_chunks(index):
    results = hybrid_search(index, "what does the onboarding process look like?")
    assert len(results) > 0
    assert all(r.fused_score is not None for r in results)
    sources = [r.chunk.source for r in results]
    assert "onboarding_guide.txt" in sources


def test_rerank_reduces_to_rerank_top_k(index):
    fused = hybrid_search(index, "data retention policy for tenant deletion")
    reranked = rerank("data retention policy for tenant deletion", fused)

    assert len(reranked) <= len(fused)
    assert all(c.rerank_score is not None for c in reranked)
    # Reranked list should be sorted descending by rerank_score.
    scores = [c.rerank_score for c in reranked]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_end_to_end_relevant_source_ranks_first(index):
    fused = hybrid_search(index, "how long is data kept before it's deleted?")
    reranked = rerank("how long is data kept before it's deleted?", fused)

    assert reranked[0].chunk.source == "data_retention_policy.txt"
