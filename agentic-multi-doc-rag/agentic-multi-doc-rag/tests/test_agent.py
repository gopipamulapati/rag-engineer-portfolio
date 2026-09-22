"""Integration tests for the heuristic agent over the real sample corpus.

These build a real index (BM25 + sentence-transformer embeddings), so they
need network access on first run to download the embedding/reranker models
-- same requirement as the retrieval tests in project 1. No OPENAI_API_KEY
is required or used here: with none set, `run_agent` always takes the
heuristic path, which is what's exercised below.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import run_agent
from app.store import build_index


@pytest.fixture(scope="module")
def index():
    return build_index()


def test_single_hop_question_uses_one_search_step(index, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = run_agent("How long is raw event data kept before deletion?", index)

    assert result.mode == "heuristic"
    assert "data_retention_policy.txt" in result.sources
    # One sub-question -> one search step + one finish step.
    assert len(result.trace) == 2


def test_multi_hop_question_covers_both_expected_documents(index, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    question = (
        "What is the default tenant rate limit on the ingest API, and how long "
        "is raw event data retained before deletion?"
    )
    result = run_agent(question, index)

    assert result.mode == "heuristic"
    assert "api_rate_limits_and_sdks.txt" in result.sources
    assert "data_retention_policy.txt" in result.sources
    # Two sub-questions -> two search steps + one finish step.
    assert len(result.trace) == 3


def test_trace_steps_are_sequential(index, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    question = (
        "Which database stores the rolled-up metrics, and how often are "
        "backups of it taken?"
    )
    result = run_agent(question, index)

    step_numbers = [t.step for t in result.trace]
    assert step_numbers == list(range(1, len(result.trace) + 1))
    assert result.trace[-1].action == "finish"
