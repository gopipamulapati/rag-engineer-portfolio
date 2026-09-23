# RAG Engineer Portfolio

Hi, I'm Gopi — this repo is a hands-on portfolio of retrieval-augmented generation (RAG) engineering projects, each one focused on a different part of building production-grade LLM systems: retrieval quality, agentic reasoning, evaluation, and conversational memory.

## Projects

### 01 - Hybrid Retrieval API (hybrid-rag-api)
A FastAPI service combining BM25 sparse search, dense embedding search, reciprocal rank fusion, and cross-encoder reranking, with a retrieval evaluation harness that measures the quality lift from each stage.

### 02 - Agentic Multi-Document RAG (agentic-multi-doc-rag)
A ReAct-style agent that decomposes multi-part questions and searches across multiple documents before answering, with a full reasoning trace and a zero-cost heuristic fallback when no LLM API key is set.

### 03 - LLM-as-Judge Evaluation Framework (coming soon)
Scoring generated RAG answers on faithfulness, relevance, and context precision/recall.

### 04 - Conversational RAG with Memory (coming soon)
Multi-turn chat over documents with session memory, streaming responses, and source citations.

## Why these projects

Each one is self-contained, with its own README, tests, and setup instructions, and runs locally with no paid API required by default. An LLM key upgrades a couple of them to smarter, dynamic behavior, but every project works and is demoable for free out of the box.
