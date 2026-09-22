"""Agent orchestration for multi-document question answering.

Two planning strategies share one execution path and one result shape, so
neither the API layer nor the evaluation script needs to know which one ran:

  - LLM-backed (`_run_llm_agent`): a ReAct-style loop where an LLM decides,
    step by step, which searches to run and when it has enough information
    to answer. This is genuine agentic tool use -- the number and content of
    searches is decided dynamically, not fixed in advance. Requires
    OPENAI_API_KEY.
  - Heuristic (`_run_heuristic_agent`): decomposes the question with the
    rules in app/decomposition.py, runs one retrieval per sub-question, and
    synthesizes an extractive answer. Zero cost, zero network, and what the
    test suite and evaluation script exercise by default.

Both call the exact same hybrid-search + rerank pipeline from
app/retrieval/ per search, so the retrieval quality is identical between
modes -- only the *planning* differs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.config import Settings, settings
from app.decomposition import decompose_question
from app.llm_backend import LLMBackend, SYSTEM_PROMPT, get_llm_backend
from app.models import ScoredChunk
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank
from app.store import RagIndex


@dataclass
class TraceStep:
    step: int
    thought: str | None
    action: str
    action_input: str
    observation: str


@dataclass
class AgentResult:
    answer: str
    mode: str  # "llm" | "heuristic"
    sources: list[str]
    trace: list[TraceStep] = field(default_factory=list)


def run_agent(question: str, index: RagIndex, cfg: Settings | None = None) -> AgentResult:
    cfg = cfg or settings
    backend = get_llm_backend(cfg)
    if backend is not None:
        return _run_llm_agent(question, index, backend, cfg)
    return _run_heuristic_agent(question, index, cfg)


def _search_tool(index: RagIndex, query: str, cfg: Settings) -> list[ScoredChunk]:
    fused = hybrid_search(index, query, cfg)
    return rerank(query, fused, cfg)


def _format_observation(chunks: list[ScoredChunk]) -> str:
    if not chunks:
        return "No relevant chunks found."
    lines = [f"- [{c.chunk.source}] {c.chunk.text[:280]}" for c in chunks[:3]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Heuristic agent: decompose -> retrieve per sub-question -> extractive merge
# ---------------------------------------------------------------------------


def _run_heuristic_agent(question: str, index: RagIndex, cfg: Settings) -> AgentResult:
    sub_questions = decompose_question(question)
    trace: list[TraceStep] = []
    all_sources: list[str] = []
    answer_parts: list[str] = []

    plan_thought = (
        f"Decomposed question into {len(sub_questions)} sub-quer"
        f"{'y' if len(sub_questions) == 1 else 'ies'}: {sub_questions}"
    )

    for i, sub_q in enumerate(sub_questions, start=1):
        top_chunks = _search_tool(index, sub_q, cfg)
        observation = _format_observation(top_chunks)

        trace.append(
            TraceStep(
                step=i,
                thought=plan_thought if i == 1 else None,
                action=f'search_documents("{sub_q}")',
                action_input=sub_q,
                observation=observation,
            )
        )

        if top_chunks:
            best = top_chunks[0]
            answer_parts.append(f"{best.chunk.text.strip()} (source: {best.chunk.source})")
            if best.chunk.source not in all_sources:
                all_sources.append(best.chunk.source)

    final_answer = "\n\n".join(answer_parts) if answer_parts else "No relevant context was found."
    trace.append(
        TraceStep(
            step=len(sub_questions) + 1,
            thought="Combined the top result from each sub-query into the final answer.",
            action="finish",
            action_input=final_answer,
            observation="",
        )
    )

    return AgentResult(answer=final_answer, mode="heuristic", sources=all_sources, trace=trace)


# ---------------------------------------------------------------------------
# LLM-backed ReAct agent
# ---------------------------------------------------------------------------


def _run_llm_agent(
    question: str, index: RagIndex, backend: LLMBackend, cfg: Settings
) -> AgentResult:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace: list[TraceStep] = []
    all_sources: list[str] = []

    for step_num in range(1, cfg.max_agent_steps + 1):
        step = backend.step(messages)

        if not step.tool_calls:
            # Model answered directly without calling `finish`.
            return AgentResult(answer=step.content or "", mode="llm", sources=all_sources, trace=trace)

        messages.append(
            {
                "role": "assistant",
                "content": step.content,
                "tool_calls": [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in step.tool_calls
                ],
            }
        )

        for tc in step.tool_calls:
            if tc.name == "finish":
                answer = tc.arguments.get("answer", "")
                trace.append(
                    TraceStep(
                        step=step_num,
                        thought=step.content,
                        action="finish",
                        action_input=answer,
                        observation="",
                    )
                )
                return AgentResult(answer=answer, mode="llm", sources=all_sources, trace=trace)

            if tc.name == "search_documents":
                query = tc.arguments.get("query", "")
                top_chunks = _search_tool(index, query, cfg)
                observation = _format_observation(top_chunks)
                for c in top_chunks:
                    if c.chunk.source not in all_sources:
                        all_sources.append(c.chunk.source)

                trace.append(
                    TraceStep(
                        step=step_num,
                        thought=step.content,
                        action=f'search_documents("{query}")',
                        action_input=query,
                        observation=observation,
                    )
                )
                messages.append({"role": "tool", "tool_call_id": tc.call_id, "content": observation})
            else:
                messages.append(
                    {"role": "tool", "tool_call_id": tc.call_id, "content": f"Unknown tool: {tc.name}"}
                )

    # Ran out of steps -- force a final answer from whatever context was gathered.
    messages.append({"role": "user", "content": "You're out of search steps. Answer now with what you have."})
    final_step = backend.step(messages)
    return AgentResult(
        answer=final_step.content or "Unable to produce a final answer within the step budget.",
        mode="llm",
        sources=all_sources,
        trace=trace,
    )
