# RAG Engineer Portfolio

[![CI](https://github.com/gopipamulapati/rag-engineer-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/gopipamulapati/rag-engineer-portfolio/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Hi, I'm Gopi, a Senior Gen AI / ML Engineer. This repo is a hands-on series of
retrieval-augmented generation (RAG) projects. Each one focuses on a different part of
building production-grade LLM systems.

Every project is self-contained, with its own README, tests, Dockerfile and evaluation
script. Each one runs locally with **no paid API by default**. An OpenAI key upgrades them to
LLM-generated answers and LLM-driven planning.

## Projects

| # | Project | What it shows | Status |
|---|---|---|---|
| 01 | [Hybrid Retrieval API](hybrid-rag-api) | BM25 + dense search, reciprocal rank fusion, cross-encoder reranking, Hit Rate@5 / MRR evaluation per stage, FastAPI | ✅ Complete |
| 02 | [Agentic Multi-Document RAG](agentic-multi-doc-rag) | ReAct-style agent that decomposes multi-hop questions, searches across documents and returns a full reasoning trace; heuristic fallback without an API key | ✅ Complete |
| 03 | [Conversational RAG with Memory](conversational-rag-langgraph) | LangGraph state graph with checkpointer memory (in-memory or SQLite), follow-up condensing, grounding self-check with retry, SSE streaming; eval shows follow-up answer hits 83% → 100% with memory | ✅ Complete |
| 04 | LLM-as-judge evaluation and guardrails | Faithfulness / relevance scoring, PII and prompt-injection detection | 🛠 Planned |

### 01: Hybrid Retrieval API
A FastAPI service that combines BM25 sparse search and dense embedding search, fuses the two
with reciprocal rank fusion, and reranks with a cross-encoder. An evaluation harness measures
Hit Rate@5 and MRR for each stage, so the quality lift from each step is measured, not assumed.

### 02: Agentic Multi-Document RAG
Multi-part questions often need facts from several documents. This agent plans which
searches to run, executes them through the same hybrid + rerank pipeline as project 01, and
returns the answer with sources and a step-by-step trace. With an API key, an LLM plans
dynamically (ReAct loop). Without one, a rule-based decomposer runs, so it's always demoable.

### 03: Conversational RAG with Memory
A multi-turn chat service built on LangGraph and LangChain. A checkpointer keeps each
session's history, follow-ups like "How long are *they* retained?" are rewritten into
standalone queries, and every answer is checked against the retrieved context before it's
returned. It's retried with a broader query or declined if it isn't grounded. Answers stream
over server-sent events, and an evaluation script measures follow-up accuracy with and
without memory.

## Running a project

```bash
cd hybrid-rag-api            # or agentic-multi-doc-rag / conversational-rag-langgraph
pip install -r requirements.txt
python scripts/ingest.py     # projects 01 and 02 only: build the index
uvicorn app.main:app --reload
pytest -q
```

## Related work

- [loan-review-agents](https://github.com/gopipamulapati/loan-review-agents): multi-agent
  mortgage pre-review with LangGraph, parallel agents, LLM guardrails, FastAPI and CI.

## License

MIT
