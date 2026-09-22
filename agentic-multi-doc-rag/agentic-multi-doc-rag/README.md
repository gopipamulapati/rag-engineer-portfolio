# Agentic Multi-Document RAG

A ReAct-style agent that decomposes multi-part questions, issues one or more
searches across a document corpus, and synthesizes a final answer citing
every source it used — with a full step-by-step reasoning trace in the API
response.

This is project 2 of a 4-part RAG engineering portfolio. Project 1
([hybrid-rag-api](../hybrid-rag-api)) answered single-hop questions well; it
had no way to notice that a question actually needs evidence from *two*
different documents, or to go get the second one. This project adds that
planning layer on top of the same retrieval core.

## The problem this solves

Ask project 1's pipeline something like *"How long does an engineer shadow
on-call before joining the primary rotation, **and** how quickly must the
on-call engineer acknowledge a page?"* and a single hybrid search has to
represent two unrelated facts — from two different documents — as one query
vector. In practice it usually retrieves chunks skewed toward whichever half
of the question dominates the embedding, and the other document never makes
it into the top-k. `scripts/evaluate_agent.py` measures this directly: a
naive single search against this project's 6 hand-labeled multi-hop
questions pulls in irrelevant documents and misses expected ones, while
decomposing the question into sub-queries first and searching each
separately reliably covers both expected sources.

## Architecture

```
question
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│                        Planner                            │
│  LLM (OpenAI, tool-calling) if OPENAI_API_KEY is set,      │
│  else a rule-based question decomposer (app/decomposition) │
└───────────────────────┬───────────────────────────────────┘
                         │ one or more search_documents(query) calls
                         ▼
        ┌───────────────────────────────────┐
        │  Same hybrid retrieval core as     │
        │  project 1: BM25 + dense + RRF +   │
        │  cross-encoder rerank              │
        └───────────────────┬───────────────┘
                             │ observation (top chunks) fed back to planner
                             ▼
              planner decides: search again, or finish
                             │
                             ▼
              final answer + sources + full trace
```

Both planners return the exact same result shape (`answer`, `mode`,
`sources`, `trace`), so the API and the evaluation script don't need to
know or care which one ran.

### LLM-backed planner (`app/agent.py::_run_llm_agent`)

A genuine ReAct loop: the LLM sees the question and two tools —
`search_documents(query)` and `finish(answer)` — and decides, turn by turn,
what to search for and when it has enough to answer. This is real agentic
behavior: the number of searches and their content are not fixed in advance,
they're decided dynamically based on what previous searches returned.
Requires `OPENAI_API_KEY`.

### Heuristic planner (`app/agent.py::_run_heuristic_agent`)

Without an API key, `app/decomposition.py` splits a compound question on
patterns like `", and how"` / `"versus"` (while deliberately *not* splitting
ordinary conjunctions like "rate limits and SDKs" that aren't two separate
asks — see the tests), runs one retrieval per sub-question, and merges the
top result from each into an extractive answer. It's a fixed, one-pass plan
rather than a dynamic loop, so it's a weaker planner than the LLM version —
but it demonstrates the same core idea (decompose, retrieve per part,
synthesize) for zero cost and zero network dependency, and it's what the
test suite runs against.

## What's included

- `app/decomposition.py` — rule-based compound-question splitting
- `app/llm_backend.py` — pluggable LLM tool-calling interface (OpenAI
  implementation); swapping in Anthropic or a local model means implementing
  one method, not touching the agent loop
- `app/agent.py` — the ReAct loop and the heuristic fallback, sharing one
  result/trace format
- `app/retrieval/`, `app/store.py`, `app/ingestion.py`, `app/text_utils.py` —
  the same hybrid retrieval core as project 1 (BM25 + dense + RRF fusion +
  cross-encoder reranking)
- `app/main.py` — FastAPI app (`/health`, `/agent/query`)
- `scripts/evaluate_agent.py` — multi-document source-coverage evaluation,
  agent vs. naive single-search baseline
- `data/multi_hop_eval_questions.json` — 6 hand-labeled questions, each
  requiring facts from exactly two documents
- `tests/test_decomposition.py` — pure-logic tests, no network required
- `tests/test_agent.py` — integration tests over the real corpus

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/ingest.py       # builds data/index/ (downloads two small
                                # models from Hugging Face on first run)

uvicorn app.main:app --reload
```

```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the default tenant rate limit on the ingest API, and how long is raw event data retained before deletion?"}'
```

Without `OPENAI_API_KEY`, this runs the heuristic planner. Set
`OPENAI_API_KEY` (and optionally `LLM_MODEL`, default `gpt-4o-mini`) to get
the dynamic LLM-driven agent loop instead — the response's `mode` field
tells you which one ran.

## Evaluating multi-document coverage

```bash
python scripts/evaluate_agent.py
```

Runs all 6 multi-hop questions through both a naive single search and the
active agent, reporting average source coverage (fraction of the expected
documents actually retrieved) for each — concrete before/after evidence for
what decomposition buys you, not just an assertion that it helps.

## Tests

```bash
pytest tests/ -v
```

## Configuration

Environment-variable overridable settings (chunking, retrieval top-k values,
model names, `MAX_AGENT_STEPS`) are in `app/config.py`.

## Running with Docker

```bash
docker build -t agentic-multi-doc-rag .
docker run -p 8000:8000 agentic-multi-doc-rag
```

## Roadmap (rest of the portfolio series)

1. Hybrid retrieval API ([project 1](../hybrid-rag-api))
2. **Multi-document agentic RAG** — this project
3. LLM-as-judge evaluation framework (this project's eval only measures
   retrieval *coverage*, not whether the generated answer is faithful to
   the retrieved context — that's next)
4. Conversational RAG with memory (multi-turn chat, streaming, session state)
