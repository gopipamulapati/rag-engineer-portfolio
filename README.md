# RAG Engineer Portfolio

Hi, I'm Gopi — this repo is a hands-on portfolio of retrieval-augmented generation (RAG) engineering projects, each one focused on a different part of building production-grade LLM systems: retrieval quality, agentic reasoning, conversational memory, answer evaluation, and document intelligence.

## Projects

### 01 - Hybrid Retrieval API (hybrid-rag-api)
A FastAPI service combining BM25 sparse search, dense embedding search, reciprocal rank fusion, and cross-encoder reranking, with a retrieval evaluation harness that measures the quality lift from each stage.

### 02 - Agentic Multi-Document RAG (agentic-multi-doc-rag)
A ReAct-style agent that decomposes multi-part questions and searches across multiple documents before answering, with a full reasoning trace and a zero-cost heuristic fallback when no LLM API key is set.

### 03 - Conversational RAG with Memory (conversational-rag-langgraph)
A multi-turn chat service built on real LangChain and LangGraph — a compiled state graph with retrieve/generate/self-check nodes, persistent session memory (in-memory or SQLite-backed), streaming responses, and an evaluation script proving follow-up questions are correctly resolved using conversation history.

### 04 - LLM-as-Judge Evaluation and Guardrails Framework (llm-judge-guardrails)
Scores generated RAG answers on faithfulness, relevance, and context precision/recall, plus a guardrails layer that detects PII and prompt-injection attempts and flags or removes unsupported claims from generated answers before they reach the user.

### 05 - Document Intelligence and Fine-Tuning (document-intelligence-finetune)
A PDF/OCR ingestion and structural partitioning pipeline for scanned and text-based documents, plus a LoRA/PEFT fine-tuning pipeline that trains a document classifier and reports a measured accuracy lift over baseline.

## Why these projects

Each one is self-contained, with its own README, tests, and setup instructions, and runs locally with no paid API required by default. An LLM key upgrades several of them to smarter, dynamic behavior, but every project works and is demoable for free out of the box.
