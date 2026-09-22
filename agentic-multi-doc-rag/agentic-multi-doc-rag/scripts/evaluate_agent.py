#!/usr/bin/env python
"""Multi-document coverage evaluation.

For each hand-labeled question in data/multi_hop_eval_questions.json (each
one requires facts from exactly two documents), this reports what fraction
of expected source documents actually got retrieved:

  - "naive single search" baseline: run the whole compound question through
    hybrid_search + rerank once, with no decomposition -- this is what you'd
    get from project 1's pipeline applied directly to a multi-hop question.
  - "agent" (whichever planner is active: LLM if OPENAI_API_KEY is set,
    otherwise the heuristic decomposition agent): the full multi-step
    pipeline in app/agent.py.

The metric is source coverage: |sources retrieved ∩ sources expected| /
|sources expected|, averaged over all questions. This is the concrete
evidence for whether decomposing multi-hop questions into sub-queries
actually pulls in evidence from documents that a single search misses.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import run_agent
from app.config import settings
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank
from app.store import load_index


def coverage(retrieved_sources: set[str], expected_sources: list[str]) -> float:
    if not expected_sources:
        return 1.0
    hit = len(retrieved_sources & set(expected_sources))
    return hit / len(expected_sources)


def main() -> None:
    index = load_index()
    eval_set = json.loads((settings.docs_dir.parent / "multi_hop_eval_questions.json").read_text())

    naive_coverages, agent_coverages = [], []
    agent_mode_used = None

    for item in eval_set:
        question = item["question"]
        expected = item["expected_sources"]

        fused = hybrid_search(index, question)
        reranked = rerank(question, fused)
        naive_sources = {c.chunk.source for c in reranked}
        naive_coverages.append(coverage(naive_sources, expected))

        result = run_agent(question, index)
        agent_mode_used = result.mode
        agent_coverages.append(coverage(set(result.sources), expected))

        print(f"Q: {question}")
        print(f"   expected:                {expected}")
        print(f"   naive single-search:     {sorted(naive_sources)}  (coverage={naive_coverages[-1]:.2f})")
        print(f"   agent ({result.mode}):   {sorted(result.sources)}  (coverage={agent_coverages[-1]:.2f})")
        print()

    n = len(eval_set)
    print(f"--- Summary over {n} multi-hop questions ---")
    print(f"Naive single-search avg source coverage: {sum(naive_coverages) / n:.2f}")
    print(f"Agent ({agent_mode_used}) avg source coverage:      {sum(agent_coverages) / n:.2f}")


if __name__ == "__main__":
    main()
