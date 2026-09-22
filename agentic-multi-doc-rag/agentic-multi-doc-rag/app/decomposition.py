"""Rule-based question decomposition for the heuristic (LLM-free) agent.

This is a small set of linguistic patterns that catch the common ways
people phrase multi-part questions -- not a learned model -- so the
heuristic agent has zero external dependencies and zero cost, and can run
fully offline in CI. It's the fallback planner exercised whenever no LLM
API key is configured (see app/agent.py): the LLM-backed agent can
decompose open-ended questions far more flexibly, but this baseline still
demonstrates genuine multi-document retrieval on the compound questions in
data/multi_hop_eval_questions.json, and it's what this repo's tests run
against since it needs no network access.

Splitting only triggers when the text *after* a connector looks like the
start of another question (starts with, or clearly contains, an
interrogative word) -- this avoids wrongly splitting ordinary conjunctions
like "rate limits and SDKs" that aren't actually two separate asks.
"""
from __future__ import annotations

import re

_INTERROGATIVE = r"(?:how|what|which|who|when|where|why|does|do|is|are)\b"

_SPLIT_RE = re.compile(
    rf",?\s+and\s+(?={_INTERROGATIVE})"
    rf"|\s+(?:compared to|versus|vs\.?)\s+(?=how|what|which|who|when|where|why)",
    flags=re.IGNORECASE,
)


def decompose_question(question: str) -> list[str]:
    """Split a compound question into independently-searchable sub-questions.

    Returns the original question unchanged, as a single-element list, when
    no split pattern matches -- so single-hop questions go through the same
    code path as multi-hop ones with no special-casing downstream.
    """
    question = question.strip()
    parts = _SPLIT_RE.split(question)
    parts = [p.strip(" ,.") for p in parts if p and p.strip(" ,.")]

    if len(parts) <= 1:
        return [question]

    return parts
