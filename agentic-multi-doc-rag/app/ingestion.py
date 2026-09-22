"""Document loading and chunking.

Chunking strategy: sentence-aware, fixed-size word windows with overlap.
We split on sentence boundaries first (rather than chopping mid-sentence)
and then greedily pack sentences into chunks up to `chunk_size_tokens`
words, carrying the trailing `chunk_overlap_tokens` words of context into
the next chunk. Word count is used as a fast, dependency-free proxy for
token count -- close enough for chunk-sizing purposes without pulling in
a tokenizer just to size windows.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.config import Settings, settings
from app.models import Chunk

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return _SENTENCE_SPLIT_RE.split(text)


def chunk_text(
    text: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[str]:
    """Pack sentences into overlapping, word-bounded chunks."""
    sentences = split_into_sentences(text)
    chunks: list[str] = []
    current_words: list[str] = []

    for sentence in sentences:
        sentence_words = sentence.split()
        if current_words and len(current_words) + len(sentence_words) > chunk_size_tokens:
            chunks.append(" ".join(current_words))
            if chunk_overlap_tokens > 0:
                current_words = current_words[-chunk_overlap_tokens:]
            else:
                current_words = []
        current_words.extend(sentence_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def load_documents(docs_dir: Path) -> dict[str, str]:
    """Load every .txt file in docs_dir. Returns {doc_id: full_text}."""
    documents: dict[str, str] = {}
    for path in sorted(docs_dir.glob("*.txt")):
        documents[path.stem] = path.read_text(encoding="utf-8")
    if not documents:
        raise FileNotFoundError(
            f"No .txt documents found in {docs_dir}. Add source documents before ingesting."
        )
    return documents


def build_chunks(docs_dir: Path, cfg: Settings | None = None) -> list[Chunk]:
    cfg = cfg or settings
    documents = load_documents(docs_dir)
    all_chunks: list[Chunk] = []

    for doc_id, text in documents.items():
        pieces = chunk_text(text, cfg.chunk_size_tokens, cfg.chunk_overlap_tokens)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{i}",
                    doc_id=doc_id,
                    source=f"{doc_id}.txt",
                    chunk_index=i,
                    text=piece,
                )
            )

    return all_chunks
