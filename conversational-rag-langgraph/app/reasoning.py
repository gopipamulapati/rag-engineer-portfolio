"""The three reasoning steps the graph runs each turn.

Each step has an LLM path and a deterministic heuristic path. The graph picks
one based on whether a chat model is configured.

  condense   follow-up question + history  ->  standalone search query
  answer     standalone query + chunks     ->  cited answer
  grounded   answer + chunks               ->  is every claim backed by the context?
"""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.retrieval import Hit, content_words, stem, tokenize

# Words that usually mean "look at the previous turn to understand me".
REFERRING = frozenset(
    "it its it's that this they them their those these there one ones he she".split()
)
FOLLOW_UP_OPENERS = ("what about", "how about", "and ", "also ", "same for")

CONDENSE_PROMPT = (
    "Rewrite the user's latest question as a standalone search query, using the "
    "conversation for any missing context (resolve pronouns like 'it' or 'that'). "
    "If it is already standalone, return it unchanged. Reply with the query only."
)
ANSWER_PROMPT = (
    "Answer the question using ONLY the context. Cite sources in square brackets, "
    "e.g. [runbook.txt]. If the context does not contain the answer, say you "
    "could not find it in the documents."
)
NOT_FOUND = "I couldn't find that in the documents."


# ---------------------------------------------------------------- condense

TOPIC_SIZE = 6  # how many topic words the heuristic condenser remembers


def is_follow_up(question: str) -> bool:
    q = question.lower().strip()
    tokens = set(tokenize(q))
    return bool(tokens & REFERRING) or q.startswith(FOLLOW_UP_OPENERS) or len(content_words(q)) <= 1


def heuristic_condense(question: str, topic: list[str] | None) -> tuple[str, list[str]]:
    """Carry the conversation topic forward when the question depends on it.

    Returns the search query and the updated topic. After
    "How long is raw event data retained?", the follow-up
    "Who can access it?" becomes "Who can access it? raw event data retained".
    The question's own words go first in the new topic, so older words fall
    off as the conversation moves on.
    """
    own = list(dict.fromkeys(content_words(question, stemmed=False)))
    if not topic or not is_follow_up(question):
        return question, own[:TOPIC_SIZE]
    own_stems = {stem(w) for w in own}
    carried = [w for w in topic if stem(w) not in own_stems]
    query = f"{question} {' '.join(carried)}".strip()
    return query, (own + carried)[:TOPIC_SIZE]


def _history_text(history: list[BaseMessage], turns: int) -> str:
    recent = history[-2 * turns :]
    lines = []
    for m in recent:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def llm_condense(llm: BaseChatModel, question: str, history: list[BaseMessage], turns: int) -> str:
    if not history:
        return question
    reply = llm.invoke(
        [
            SystemMessage(CONDENSE_PROMPT),
            HumanMessage(f"Conversation:\n{_history_text(history, turns)}\n\nLatest: {question}"),
        ]
    )
    text = str(reply.content).strip()
    return text or question


# ---------------------------------------------------------------- answer

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def heuristic_answer(question: str, query: str, hits: list[Hit], max_sentences: int = 2) -> str:
    """Extractive answer: the best-matching sentences from the top chunks."""
    """Extractive answer: the best-matching sentences from the top chunks.

    Words from the user's own question count double compared with topic
    words carried over from earlier turns, so the new part of a follow-up
    decides which sentence is picked.
    """
    if not hits:
        return NOT_FOUND
    own = set(content_words(question))
    terms = set(content_words(query))
    scored: list[tuple[float, int, str, str]] = []
    for rank, hit in enumerate(hits):
        for pos, sentence in enumerate(_SENTENCE.split(hit.chunk.text)):
            if len(sentence.split()) < 5:  # skip heading-like fragments
                continue
            words = set(content_words(sentence))
            score = 2 * len(own & words) + len((terms - own) & words)
            if score:
                # Prefer higher-ranked chunks, then earlier sentences.
                scored.append((score - 0.5 * rank, -pos, sentence, hit.chunk.source))
    if not scored:
        return NOT_FOUND
    # If any sentence mentions the user's own words, only consider those.
    on_topic = [row for row in scored if own & set(content_words(row[2]))]
    ranked = sorted(on_topic or scored, reverse=True)
    best = [ranked[0]] + [r for r in ranked[1:max_sentences] if r[0] >= 0.75 * ranked[0][0]]
    return " ".join(f"{s} [{src}]" for _, _, s, src in best)


def _context_text(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{h.chunk.source}] {h.chunk.text}" for h in hits)


def llm_answer(llm: BaseChatModel, query: str, hits: list[Hit], history: list[BaseMessage]) -> str:
    if not hits:
        return NOT_FOUND
    messages: list[BaseMessage] = [SystemMessage(ANSWER_PROMPT)]
    messages += history[-4:]
    messages.append(HumanMessage(f"Context:\n{_context_text(hits)}\n\nQuestion: {query}"))
    reply: AIMessage = llm.invoke(messages)
    return str(reply.content).strip()


# ---------------------------------------------------------------- self-check

_CITATION = re.compile(r"\[[^\]]+\]")


def grounding_score(
    answer: str, hits: list[Hit], question: str | None = None, topic: list[str] | None = None
) -> float:
    """How well the answer is supported by the retrieved context (0 to 1).

    Two checks:
      * relevance: at least one of the user's own question words (or, for a
        follow-up, one conversation-topic word) must appear in the context,
        otherwise the retriever found something off-topic
      * support: the share of the answer's content words found in the context
    """
    if answer == NOT_FOUND:
        return 1.0  # declining to answer is always grounded
    if not hits:
        return 0.0
    context = set(content_words(_context_text(hits)))
    own = set(content_words(question or ""))
    if own and not own & context and not (topic and set(content_words(" ".join(topic))) & context):
        return 0.0
    words = content_words(_CITATION.sub(" ", answer))
    if not words:
        return 0.0
    return sum(w in context for w in words) / len(words)
