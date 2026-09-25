"""LangGraph state graph for multi-turn RAG with persistent session memory.

    START -> condense -> retrieve -> generate -> self_check --grounded--> finalize -> END
                            ^                        |
                            +---- retry (broader) ---+  (up to MAX_RETRIEVAL_ATTEMPTS)

Memory comes from a LangGraph checkpointer keyed by ``thread_id`` (the session
id). The message history and the running topic persist between
calls, so the next question can use them. Use ``InMemorySaver`` for development
and ``SqliteSaver`` for sessions that survive a restart.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app import reasoning
from app.config import Settings, settings
from app.retrieval import Chunk, Hit, Retriever, content_words, load_chunks


class ChatState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # full conversation (persisted)
    question: str  # this turn's raw user question
    query: str  # standalone search query for this turn
    topic: list[str]  # running topic words (persisted, used by the heuristic condenser)
    hits: list[dict[str, Any]]  # retrieved chunks (plain dicts so they serialize)
    answer: str
    grounding: float
    attempts: int


def _to_hits(rows: list[dict[str, Any]]) -> list[Hit]:
    return [Hit(Chunk(r["source"], r["text"]), r["score"]) for r in rows]


def build_graph(
    retriever: Retriever | None = None,
    llm: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    cfg: Settings = settings,
):
    retriever = retriever or Retriever(load_chunks(cfg.docs_dir))
    checkpointer = checkpointer or InMemorySaver()

    def condense(state: ChatState) -> dict:
        question = state["question"]
        history = state["messages"][:-1]  # everything before this turn
        if llm is not None:
            query = reasoning.llm_condense(llm, question, history, cfg.history_turns)
            topic = content_words(query, stemmed=False)[: reasoning.TOPIC_SIZE]
        else:
            query, topic = reasoning.heuristic_condense(question, state.get("topic"))
        return {"query": query, "topic": topic, "attempts": 0}

    def retrieve(state: ChatState) -> dict:
        attempts = state.get("attempts", 0) + 1
        query = state["query"]
        if attempts > 1:
            # Retry with a broader query: add terms from recent user turns.
            recent = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
            extra = content_words(" ".join(str(t) for t in recent[-cfg.history_turns :]))
            query = f"{query} {' '.join(dict.fromkeys(extra))}"
        # The user's own words appear twice so they outweigh carried topic words.
        hits = retriever.search(f"{state['question']} {query}", cfg.top_k)
        rows = [{"source": h.chunk.source, "text": h.chunk.text, "score": h.score} for h in hits]
        return {"hits": rows, "attempts": attempts}

    def generate(state: ChatState) -> dict:
        hits = _to_hits(state["hits"])
        if llm is not None:
            history = state["messages"][:-1]
            answer = reasoning.llm_answer(llm, state["query"], hits, history)
        else:
            answer = reasoning.heuristic_answer(state["question"], state["query"], hits)
        return {"answer": answer}

    def self_check(state: ChatState) -> dict:
        topic = state.get("topic") if reasoning.is_follow_up(state["question"]) else None
        score = reasoning.grounding_score(
            state["answer"], _to_hits(state["hits"]), state["question"], topic
        )
        return {"grounding": score}

    def route_after_check(state: ChatState) -> str:
        if state["grounding"] >= cfg.grounding_threshold:
            return "finalize"
        if state["attempts"] < cfg.max_retrieval_attempts:
            return "retrieve"
        return "decline"

    def decline(state: ChatState) -> dict:
        # Never return an answer that failed the grounding check.
        return {"answer": reasoning.NOT_FOUND, "hits": []}

    def finalize(state: ChatState) -> dict:
        return {"messages": [AIMessage(state["answer"])]}

    g = StateGraph(ChatState)
    for name, fn in [
        ("condense", condense),
        ("retrieve", retrieve),
        ("generate", generate),
        ("self_check", self_check),
        ("decline", decline),
        ("finalize", finalize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "condense")
    g.add_edge("condense", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "self_check")
    g.add_conditional_edges("self_check", route_after_check, ["finalize", "retrieve", "decline"])
    g.add_edge("decline", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def turn_input(question: str) -> dict:
    return {"messages": [HumanMessage(question)], "question": question}


def session_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def ask(graph, session_id: str, question: str) -> dict:
    """Run one conversational turn and return a JSON-friendly result."""
    state = graph.invoke(turn_input(question), session_config(session_id))
    return {
        "answer": state["answer"],
        "standalone_query": state["query"],
        "sources": list(dict.fromkeys(h["source"] for h in state.get("hits", []))),
        "grounding": round(state["grounding"], 3),
        "retrieval_attempts": state["attempts"],
    }
