#!/usr/bin/env python
"""Build the hybrid index from data/sample_docs and persist it to data/index.

Run this whenever documents change, then (re)start the API to pick up the
new index:

    python scripts/ingest.py
    uvicorn app.main:app --reload
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.store import build_index, save_index


def main() -> None:
    print(f"Loading documents from {settings.docs_dir} ...")
    start = time.time()

    index = build_index()

    print(f"Built {index.size} chunks in {time.time() - start:.1f}s")
    save_index(index, settings.index_dir)
    print(f"Saved index to {settings.index_dir}")


if __name__ == "__main__":
    main()
