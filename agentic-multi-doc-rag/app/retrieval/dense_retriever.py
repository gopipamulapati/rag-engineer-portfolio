"""Dense (embedding) retrieval — good at semantic/paraphrase matches that
don't share exact keywords with the query (e.g. "how fast can we recover
from a regional outage?" matching failover procedure text)."""
from __future__ import annotations

import numpy as np

from app.models import Chunk
from app.store import RagIndex
from app.text_utils import normalize_vectors


def dense_search(index: RagIndex, query: str, top_k: int) -> list[tuple[Chunk, float]]:
    query_embedding = index.embedder.encode([query], convert_to_numpy=True)
    query_embedding = normalize_vectors(query_embedding)[0]

    # Cosine similarity == dot product, since both sides are L2-normalized.
    similarities = index.embeddings @ query_embedding

    ranked_indices = np.argsort(similarities)[::-1][:top_k]
    results = [(index.chunks[i], float(similarities[i])) for i in ranked_indices]
    return results
