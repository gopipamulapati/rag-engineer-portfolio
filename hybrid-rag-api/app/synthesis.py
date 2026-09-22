"""Turn retrieved chunks into a final answer.

If an OpenAI API key is configured, the retrieved chunks are passed to an
LLM with an instruction to answer using only that context and to cite
sources by filename. Without a key, the service still returns a useful,
fully deterministic result: an extractive "answer" made of the single
highest-scoring chunk, so the API is usable and demoable with zero external
dependencies or cost.
"""
from __future__ import annotations

import os

from app.config import Settings, settings
from app.models import ScoredChunk

SYSTEM_PROMPT = (
    "You are a precise assistant answering questions using only the provided "
    "context. If the context does not contain the answer, say so explicitly "
    "instead of guessing. Cite the source filename(s) you used in parentheses "
    "at the end of relevant sentences."
)


def _format_context(chunks: list[ScoredChunk]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[Source: {c.chunk.source}]\n{c.chunk.text}")
    return "\n\n---\n\n".join(blocks)


def synthesize_answer(
    question: str, chunks: list[ScoredChunk], cfg: Settings | None = None
) -> tuple[str | None, str]:
    """Returns (answer_text, mode) where mode is 'llm' or 'extractive'."""
    cfg = cfg or settings

    if not chunks:
        return "No relevant context was found for this question.", "extractive"

    if cfg.openai_api_key:
        try:
            return _llm_answer(question, chunks, cfg), "llm"
        except Exception as exc:  # pragma: no cover - network/SDK failure path
            fallback = _extractive_answer(chunks)
            return f"{fallback}\n\n[LLM synthesis unavailable: {exc}]", "extractive"

    return _extractive_answer(chunks), "extractive"


def _extractive_answer(chunks: list[ScoredChunk]) -> str:
    best = chunks[0]
    return f"{best.chunk.text.strip()} (source: {best.chunk.source})"


def _llm_answer(question: str, chunks: list[ScoredChunk], cfg: Settings) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=cfg.openai_api_key)
    context = _format_context(chunks)

    response = client.chat.completions.create(
        model=cfg.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()
