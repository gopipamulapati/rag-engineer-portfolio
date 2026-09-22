#!/usr/bin/env python
"""Lightweight retrieval evaluation harness.

This is a smaller, self-contained cousin of the standalone LLM-judge
evaluation project: rather than judging generated answers, it measures
whether the *retrieval* stage surfaces the right source document, using a
small hand-labeled question set (data/eval_questions.json). It reports two
standard IR metrics per pipeline stage:

  - Hit Rate @ k: fraction of questions where the expected source document
    appears anywhere in the top-k retrieved chunks.
  - MRR (Mean Reciprocal Rank): rewards ranking the right source higher,
    not just including it somewhere in the top k.

It runs this for BM25 alone, dense alone, and the final hybrid+reranked
pipeline, so you can see the concrete lift each stage contributes --
exactly the kind of evidence worth putting in a README or a resume bullet.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.retrieval.bm25_retriever import bm25_search
from app.retrieval.dense_retriever import dense_search
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank
from app.store import load_index

TOP_K = 5


def hit_rate_and_mrr(results_per_question: list[list[str]], expected: list[str]) -> tuple[float, float]:
    hits = 0
    reciprocal_ranks = []
    for sources, expected_source in zip(results_per_question, expected):
        if expected_source in sources[:TOP_K]:
            hits += 1
        if expected_source in sources:
            rank = sources.index(expected_source) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    n = len(expected)
    return hits / n, sum(reciprocal_ranks) / n


def main() -> None:
    index = load_index()
    eval_set = json.loads((settings.docs_dir.parent / "eval_questions.json").read_text())

    bm25_sources, dense_sources, hybrid_sources = [], [], []
    expected_sources = [item["expected_source"] for item in eval_set]

    for item in eval_set:
        question = item["question"]

        bm25_results = bm25_search(index, question, top_k=10)
        bm25_sources.append([c.source for c, _ in bm25_results])

        dense_results = dense_search(index, question, top_k=10)
        dense_sources.append([c.source for c, _ in dense_results])

        fused = hybrid_search(index, question)
        reranked = rerank(question, fused)
        hybrid_sources.append([c.chunk.source for c in reranked])

    print(f"Evaluating {len(eval_set)} questions against {index.size} indexed chunks\n")

    for name, sources in [
        ("BM25 only", bm25_sources),
        ("Dense only", dense_sources),
        ("Hybrid + rerank", hybrid_sources),
    ]:
        hit_rate, mrr = hit_rate_and_mrr(sources, expected_sources)
        print(f"{name:18s}  Hit Rate@{TOP_K}: {hit_rate:.2f}   MRR: {mrr:.2f}")


if __name__ == "__main__":
    main()
