import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DOCS_DIR  # noqa: E402
from app.retrieval import Retriever, load_chunks  # noqa: E402


@pytest.fixture(scope="session")
def retriever():
    return Retriever(load_chunks(DOCS_DIR))
