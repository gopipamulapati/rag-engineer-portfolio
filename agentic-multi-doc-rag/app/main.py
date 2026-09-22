"""FastAPI service exposing the multi-document agent."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.agent import run_agent
from app.config import settings
from app.llm_backend import get_llm_backend
from app.models import AgentQueryRequest, AgentQueryResponse, HealthResponse, TraceStepResponse
from app.store import RagIndex, load_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentic_multi_doc_rag")

_state: dict[str, RagIndex] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading RAG index from %s", settings.index_dir)
    try:
        _state["index"] = load_index()
        logger.info("Loaded index with %d chunks", _state["index"].size)
    except FileNotFoundError as exc:
        logger.warning("%s Starting without an index -- run `python scripts/ingest.py`.", exc)
        _state["index"] = None
    yield
    _state.clear()


app = FastAPI(
    title="Agentic Multi-Document RAG",
    description=(
        "A ReAct-style agent that decomposes multi-part questions and issues "
        "one or more searches across a document corpus before synthesizing a "
        "final answer, with an LLM-free heuristic fallback."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _get_index() -> RagIndex:
    index = _state.get("index")
    if index is None:
        raise HTTPException(
            status_code=503,
            detail="Index not built yet. Run `python scripts/ingest.py` and restart the service.",
        )
    return index


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    index = _state.get("index")
    return HealthResponse(
        status="ok" if index is not None else "index_not_built",
        num_chunks_indexed=index.size if index is not None else 0,
        agent_mode="llm" if get_llm_backend() is not None else "heuristic",
        embedding_model=settings.embedding_model_name,
        reranker_model=settings.reranker_model_name,
    )


@app.post("/agent/query", response_model=AgentQueryResponse)
def agent_query(request: AgentQueryRequest) -> AgentQueryResponse:
    index = _get_index()
    result = run_agent(request.question, index)

    return AgentQueryResponse(
        question=request.question,
        answer=result.answer,
        mode=result.mode,
        sources=result.sources,
        trace=[
            TraceStepResponse(
                step=t.step,
                thought=t.thought,
                action=t.action,
                action_input=t.action_input,
                observation=t.observation,
            )
            for t in result.trace
        ],
    )
