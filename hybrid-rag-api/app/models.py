"""Pydantic schemas shared between the API layer and retrieval internals."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    chunk_index: int
    text: str


class ScoredChunk(BaseModel):
    chunk: Chunk
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to return")
    synthesize_answer: bool = Field(
        True, description="If true, generate a natural-language answer with citations"
    )


class RetrievedChunkResponse(BaseModel):
    doc_id: str
    source: str
    chunk_index: int
    text: str
    bm25_score: Optional[float]
    dense_score: Optional[float]
    fused_score: Optional[float]
    rerank_score: Optional[float]


class QueryResponse(BaseModel):
    question: str
    answer: Optional[str]
    answer_mode: str  # "llm" | "extractive"
    retrieved_chunks: list[RetrievedChunkResponse]


class HealthResponse(BaseModel):
    status: str
    num_chunks_indexed: int
    embedding_model: str
    reranker_model: str
