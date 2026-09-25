"""FastAPI service: multi-turn chat with session memory and streaming."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from app.config import Settings, settings
from app.graph import ask, build_graph, session_config, turn_input
from app.llm import get_chat_model


def make_checkpointer(cfg: Settings) -> BaseCheckpointSaver:
    if cfg.memory_backend == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(cfg.sqlite_path, check_same_thread=False)
        return SqliteSaver(conn)
    return InMemorySaver()


checkpointer = make_checkpointer(settings)
llm = get_chat_model(settings)
graph = build_graph(llm=llm, checkpointer=checkpointer)

app = FastAPI(
    title="Conversational RAG (LangGraph)",
    description="Multi-turn RAG with session memory, self-checking and streaming.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    standalone_query: str
    sources: list[str]
    grounding: float
    retrieval_attempts: int


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "mode": "llm" if llm is not None else "heuristic",
        "memory_backend": settings.memory_backend,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> dict:
    return ask(graph, req.session_id, req.message)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream(session_id: str, message: str) -> Iterator[str]:
    """Server-sent events: one `step` per graph node, `token` events while an
    LLM writes the answer, then a final `answer` event."""
    streamed_tokens = False
    for mode, payload in graph.stream(
        turn_input(message), session_config(session_id), stream_mode=["updates", "messages"]
    ):
        if mode == "updates":
            for node in payload:
                yield _sse("step", {"node": node})
        elif mode == "messages":
            chunk, meta = payload
            if meta.get("langgraph_node") == "generate" and isinstance(chunk, AIMessageChunk):
                if chunk.content:
                    streamed_tokens = True
                    yield _sse("token", {"text": chunk.content})
    state = graph.get_state(session_config(session_id)).values
    if not streamed_tokens:
        # Heuristic mode has no LLM tokens, so stream the final answer word by word.
        for word in state["answer"].split(" "):
            yield _sse("token", {"text": word + " "})
    yield _sse(
        "answer",
        {
            "answer": state["answer"],
            "standalone_query": state["query"],
            "sources": list(dict.fromkeys(h["source"] for h in state.get("hits", []))),
        },
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(_stream(req.session_id, req.message), media_type="text/event-stream")


@app.get("/sessions/{session_id}/history")
def history(session_id: str) -> dict:
    values = graph.get_state(session_config(session_id)).values
    if not values:
        raise HTTPException(status_code=404, detail="Unknown session")
    return {
        "session_id": session_id,
        "messages": [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in values.get("messages", [])
        ],
    }


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    checkpointer.delete_thread(session_id)
