"""Small shared helpers used by both index building and retrieval, kept in
one place so BM25 tokenization and embedding normalization stay identical
between build time and query time."""
from __future__ import annotations

import numpy as np


def tokenize_for_bm25(text: str) -> list[str]:
    return text.lower().split()


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vectors / norms
