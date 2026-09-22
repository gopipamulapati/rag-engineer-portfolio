"""Pydantic schemas shared between the API layer and internals."""
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


class AgentQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class TraceStepResponse(BaseModel):
    step: int
    thought: Optional[str]
    action: str
    action_input: str
    observation: str


class AgentQueryResponse(BaseModel):
    question: str
    answer: str
    mode: str  # "llm" | "heuristic"
    sources: list[str]
    trace: list[TraceStepResponse]


class HealthResponse(BaseModel):
    status: str
    num_chunks_indexed: int
    agent_mode: str  # "llm" | "heuristic" -- which planner is active
    embedding_model: str
    reranker_model: str
