# Hybrid RAG API

A retrieval-augmented Q&A service that combines **BM25 sparse search**,
**dense embedding search**, **reciprocal rank fusion**, and **cross-encoder
reranking** — served as a FastAPI app, with an evaluation harness that
measures the retrieval quality lift each stage contributes.

This is project 1 of a 4-part RAG engineering portfolio. It focuses on
retrieval quality specifically: getting the *right* passages in front of an
LLM (or a human) before worrying about agentic orchestration, evaluation of
generated answers, or conversational memory — those are separate projects
in the series.

## Why hybrid retrieval

Dense embedding search is good at semantic/paraphrase matches ("how do we
recover if a region goes down?" → failover procedure) but can blur on exact
keywords, numbers, and specific identifiers. BM25 is the opposite: excellent
at exact term matches ("5,000 events per second") but blind to paraphrase.
Using only one of them leaves a systematic gap. This project runs both in
parallel and fuses their rankings.

## Architecture

```
                    ┌─────────────────┐
   question ──────▶ │   /query (API)  │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
      ┌───────────────┐              ┌───────────────┐
      │  BM25 search   │              │ Dense search   │
      │ (sparse, exact │              │ (embeddings,   │
      │  keyword match)│              │ semantic match)│
      └───────┬────────┘              └───────┬────────┘
              │        top-20 each             │
              └───────────────┬────────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │ Reciprocal Rank Fusion   │
                  │ (scale-free rank blend)  │
                  └────────────┬────────────┘
                               │ top-10
                               ▼
                  ┌─────────────────────────┐
                  │  Cross-encoder reranker  │
                  │ (query+chunk pair score) │
                  └────────────┬────────────┘
                               │ top-k (default 5)
                               ▼
                  ┌─────────────────────────┐
                  │  Answer synthesis        │
                  │  (LLM if key set, else   │
                  │   extractive fallback)   │
                  └────────────┬────────────┘
                               ▼
                          JSON response
```

**Why reciprocal rank fusion instead of averaging scores?** BM25 scores and
cosine similarities live on incomparable scales — averaging them lets
whichever retriever happens to produce bigger numbers dominate. RRF only
uses each result's *rank position*, so it combines the two lists without
needing a hand-tuned blending weight.

**Why a cross-encoder rerank step?** The bi-encoder used for dense search
embeds the query and each chunk independently (fast, scales to large
corpora, but query and chunk never directly interact). A cross-encoder
scores a `(query, chunk)` pair jointly, which is much more accurate but too
slow to run over an entire corpus — so it's applied only to the ~10
candidates that already survived fusion, as a final precision pass.

## What's included

- `app/ingestion.py` — sentence-aware chunking with configurable size/overlap
- `app/store.py` — builds and persists the BM25 index + dense embeddings
- `app/retrieval/` — BM25 retriever, dense retriever, RRF fusion, cross-encoder reranker
- `app/synthesis.py` — LLM answer synthesis with citation instructions, or a
  deterministic extractive fallback when no API key is configured (the
  service is fully usable and demoable with zero external cost)
- `app/main.py` — FastAPI app (`/health`, `/query`)
- `scripts/ingest.py` — builds the index from `data/sample_docs/`
- `scripts/evaluate.py` — retrieval evaluation harness (see below)
- `tests/` — unit + integration tests over the real sample corpus
- `data/sample_docs/` — a small original set of fictional company engineering
  docs (architecture, incident response, data retention, onboarding, API
  limits) used as the demo corpus — swap in your own `.txt` files here
- `data/eval_questions.json` — 8 hand-labeled questions with expected source
  documents, used by `scripts/evaluate.py`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/ingest.py         # builds data/index/ (downloads two small
                                  # models from Hugging Face on first run)

uvicorn app.main:app --reload
```

Then query it:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How long is raw event data kept before deletion?", "top_k": 3}'
```

Without an `OPENAI_API_KEY` set, `/query` returns the single highest-scoring
chunk as an extractive answer. Set `OPENAI_API_KEY` (and optionally
`LLM_MODEL`, default `gpt-4o-mini`) to get a synthesized natural-language
answer with source citations instead.

## Evaluating retrieval quality

```bash
python scripts/evaluate.py
```

This runs the 8 questions in `data/eval_questions.json` through BM25 alone,
dense search alone, and the full hybrid+rerank pipeline, reporting **Hit
Rate@5** and **MRR** for each — concrete evidence of what hybrid retrieval
and reranking actually buy you over either method alone, rather than just
asserting it.

## Configuration

All tunables (`chunk_size_tokens`, `bm25_top_k`, `dense_top_k`,
`rerank_top_k`, `rrf_k`, model names, etc.) are environment-variable
overridable — see `app/config.py`.

## Running with Docker

```bash
docker build -t hybrid-rag-api .
docker run -p 8000:8000 hybrid-rag-api
```

## Tests

```bash
pytest tests/ -v
```

## Using your own documents

Drop `.txt` files into `data/sample_docs/`, update `data/eval_questions.json`
with a few representative questions and their expected source files, then
re-run `python scripts/ingest.py`.

## Roadmap (rest of the portfolio series)

1. **Hybrid retrieval API** — this project
2. Multi-document agentic RAG (multi-hop reasoning, tool-using agent)
3. LLM-as-judge evaluation framework (faithfulness, relevance, context
   precision/recall over generated answers — this project's `evaluate.py`
   only measures retrieval, not generation quality)
4. Conversational RAG with memory (multi-turn chat, streaming, session state)
