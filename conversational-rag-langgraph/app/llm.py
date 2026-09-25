"""Optional LangChain chat model.

The graph accepts any LangChain ``BaseChatModel``. It uses ``ChatOpenAI`` when
``OPENAI_API_KEY`` is set. Otherwise it returns None, and the graph uses its
deterministic heuristics. Tests pass LangChain's fake chat models to exercise
the LLM code paths offline, including token streaming.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.config import Settings, settings


def get_chat_model(cfg: Settings = settings) -> BaseChatModel | None:
    if not cfg.openai_api_key:
        return None
    from langchain_openai import ChatOpenAI  # optional dependency

    return ChatOpenAI(model=cfg.openai_model, api_key=cfg.openai_api_key, temperature=0)
