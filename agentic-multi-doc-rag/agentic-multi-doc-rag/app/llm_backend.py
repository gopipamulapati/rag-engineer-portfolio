"""Pluggable LLM backend for the agent's planning / tool-calling step.

Only an OpenAI implementation ships today (OpenAI's function-calling format
is the most widely supported one to demonstrate), but `app/agent.py` only
depends on the `LLMBackend` protocol below -- adding an Anthropic or local
open-weights backend later means implementing one method (`step`), not
touching the agent loop itself.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, settings

SYSTEM_PROMPT = (
    "You are a research agent answering questions about Northwind Analytics' "
    "internal engineering documentation. You do not have the documents "
    "memorized: you must call search_documents to look up facts, once per "
    "distinct piece of information you still need. Many questions require "
    "multiple searches across different documents before you have enough to "
    "answer -- do not guess, and do not stop after one search if the "
    "question has multiple parts. When you have enough information, call "
    "finish with a complete answer that cites the source filename(s) in "
    "parentheses for each fact."
)

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the document corpus for chunks relevant to one specific, "
                "focused query. Call this once per distinct piece of information "
                "you still need -- do not cram a multi-part question into one search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A focused search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call once you have enough information to fully answer the original question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer, citing source filenames in parentheses",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass
class LLMStep:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMBackend(ABC):
    @abstractmethod
    def step(self, messages: list[dict[str, Any]]) -> LLMStep:
        """Given the running conversation, return the model's next thought and/or tool call(s)."""


class OpenAIBackend(LLMBackend):
    def __init__(self, cfg: Settings):
        from openai import OpenAI

        self._client = OpenAI(api_key=cfg.openai_api_key)
        self._model = cfg.llm_model

    def step(self, messages: list[dict[str, Any]]) -> LLMStep:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            temperature=0.0,
        )
        choice = response.choices[0].message
        tool_calls = [
            ToolCall(
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}"),
                call_id=tc.id,
            )
            for tc in (choice.tool_calls or [])
        ]
        return LLMStep(content=choice.content, tool_calls=tool_calls)


def get_llm_backend(cfg: Settings | None = None) -> LLMBackend | None:
    """Returns an LLM backend if an API key is configured, else None so the
    caller falls back to the heuristic agent."""
    cfg = cfg or settings
    if cfg.openai_api_key:
        return OpenAIBackend(cfg)
    return None
