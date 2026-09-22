import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.decomposition import decompose_question


def test_single_hop_question_is_not_split():
    q = "How long is raw event data kept before deletion?"
    assert decompose_question(q) == [q]


def test_conjunction_without_second_question_is_not_split():
    # "and" here joins two nouns, not two questions -- must not be split.
    q = "What are the API rate limits and SDKs available?"
    assert decompose_question(q) == [q]


def test_compound_question_with_and_how_is_split():
    q = (
        "How long does an engineer shadow on-call before joining the primary "
        "rotation, and how quickly must the on-call engineer acknowledge a page?"
    )
    parts = decompose_question(q)
    assert len(parts) == 2
    assert "shadow on-call" in parts[0]
    assert "acknowledge a page" in parts[1]


def test_compound_question_with_and_what_is_split():
    q = "What is the default tenant rate limit, and what is the raw event retention period?"
    parts = decompose_question(q)
    assert len(parts) == 2


def test_versus_pattern_is_split():
    q = "What does BM25 retrieve versus what does dense search retrieve for this query?"
    parts = decompose_question(q)
    assert len(parts) == 2
