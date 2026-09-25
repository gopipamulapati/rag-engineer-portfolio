"""Paragraph chunking and BM25 retrieval.

Project 01 (hybrid-rag-api) covers dense + sparse fusion and reranking. This
project focuses on conversation, so it uses a light BM25 index that builds in
milliseconds. ``Retriever.search`` is the only interface the graph depends on,
so the hybrid pipeline from project 01 can be swapped in without touching
the graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in is it its
    me my of on or our so that the their them then there these they this those to
    was we what when where which who why will with you your about into than also
    any all after before under over per each more most other some such only own
    same just should would could may might must shall not no nor too very tell
    please explain describe much many long often usually quickly isn aren doesn
    don didn wasn can't happens happen get got""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def stem(word: str) -> str:
    """Tiny suffix stripper so "backs"/"backed" and "acknowledge"/"acknowledged" match."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    for suffix, min_len in (("ing", 6), ("ed", 5), ("es", 5), ("s", 4)):
        if word.endswith(suffix) and len(word) >= min_len and not word.endswith("ss"):
            word = word[: -len(suffix)]
            break
    return word[:-1] if word.endswith("e") and len(word) > 4 else word


def content_words(text: str, stemmed: bool = True) -> list[str]:
    """Lower-cased non-stopword tokens, stemmed by default for matching."""
    words = [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]
    return [stem(w) for w in words] if stemmed else words


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def load_chunks(docs_dir: Path) -> list[Chunk]:
    """One chunk per paragraph; the document title line is skipped."""
    chunks: list[Chunk] = []
    for path in sorted(Path(docs_dir).glob("*.txt")):
        paragraphs = re.split(r"\n\s*\n", path.read_text(encoding="utf-8"))
        for para in paragraphs[1:] if len(paragraphs) > 1 else paragraphs:
            text = " ".join(para.split())
            if text:
                chunks.append(Chunk(source=path.name, text=text))
    if not chunks:
        raise ValueError(f"No .txt documents found in {docs_dir}")
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([content_words(c.text) for c in chunks])

    def search(self, query: str, k: int = 3) -> list[Hit]:
        terms = content_words(query)
        if not terms:
            return []
        scores = self._bm25.get_scores(terms)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]
