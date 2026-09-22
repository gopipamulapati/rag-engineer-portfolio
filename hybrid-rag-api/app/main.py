"""FastAPI service exposing the hybrid RAG pipeline.

The index is loaded once at startup (see `lifespan`) rather than per
request, since embedding-model loading and index loading are relatively
expensive and the index is read-only during normal operation.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.models import HealthResponse, QueryRequest, QueryResponse, RetrievedChunkResponse
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank
from app.store import RagIndex, load_index
from app.synthesis import synthesize_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hybrid_rag_api")

_state: dict[str, RagIndex] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading RAG index from %s", settings.index_dir)
    try:
        _state["index"] = load_index()
        logger.info("Loaded index with %d chunks", _state["index"].size)
    except FileNotFoundError as exc:
        logger.warning(
            "%s Starting without an index -- /query will return 503 until "
            "`python scripts/ingest.py` has been run.",
            exc,
        )
        _state["index"] = None
    yield
    _state.clear()


app = FastAPI(
    title="Hybrid RAG API",
    description=(
        "Retrieval-augmented Q&A over a document set, combining BM25 sparse "
        "search, dense embedding search, reciprocal rank fusion, and "
        "cross-encoder reranking."
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
        embedding_model=settings.embedding_model_name,
        reranker_model=settings.reranker_model_name,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    index = _get_index()

    fused_candidates = hybrid_search(index, request.question)
    reranked = rerank(request.question, fused_candidates)
    top_chunks = reranked[: request.top_k]

    answer, mode = (
        synthesize_answer(request.question, top_chunks)
        if request.synthesize_answer
        else (None, "none")
    )

    return QueryResponse(
        question=request.question,
        answer=answer,
        answer_mode=mode,
        retrieved_chunks=[
            RetrievedChunkResponse(
                doc_id=c.chunk.doc_id,
                source=c.chunk.source,
                chunk_index=c.chunk.chunk_index,
                text=c.chunk.text,
                bm25_score=c.bm25_score,
                dense_score=c.dense_score,
                fused_score=c.fused_score,
                rerank_score=c.rerank_score,
            )
            for c in top_chunks
        ],
    )
