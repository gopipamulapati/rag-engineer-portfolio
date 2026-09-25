"""Measure what conversation memory buys on follow-up questions.

Every conversation in data/conversations.json runs twice:
  * with memory:    all turns share one session (thread_id)
  * without memory: every turn is sent to a fresh, empty session

For each turn we check:
  * source hit: the expected document is the top-ranked source
  * answer hit: the expected fact (e.g. "13 months") appears in the answer

This is a small, hand-written smoke test (18 turns over 5 documents), not a
benchmark. The heuristic mode was developed against it, so treat its numbers
as a regression check and a demonstration of the memory effect.

Usage:  python scripts/evaluate_memory.py
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import ask, build_graph  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "conversations.json"


def run(conversations: list[dict], use_memory: bool) -> dict:
    graph = build_graph()
    totals = {"first": [0, 0, 0], "follow_up": [0, 0, 0]}  # [turns, source hits, answer hits]
    for conv in conversations:
        session = f"{conv['id']}-{uuid.uuid4()}"
        for i, turn in enumerate(conv["turns"]):
            sid = session if use_memory else f"{session}-{i}"
            result = ask(graph, sid, turn["question"])
            bucket = totals["first" if i == 0 else "follow_up"]
            bucket[0] += 1
            bucket[1] += result["sources"][:1] == [turn["source"]]
            bucket[2] += turn["expect"].lower() in result["answer"].lower()
    return totals


def pct(hits: int, n: int) -> str:
    return f"{hits / n:.0%}" if n else "-"


def main() -> None:
    conversations = json.loads(DATA.read_text(encoding="utf-8"))
    with_mem = run(conversations, use_memory=True)
    without = run(conversations, use_memory=False)

    print(f"{len(conversations)} conversations, {with_mem['follow_up'][0]} follow-up turns\n")
    print(f"{'turn type':<12}{'metric':<14}{'no memory':>11}{'with memory':>13}")
    for kind in ("first", "follow_up"):
        n = with_mem[kind][0]
        for idx, name in ((1, "source hit"), (2, "answer hit")):
            print(
                f"{kind:<12}{name:<14}{pct(without[kind][idx], n):>11}"
                f"{pct(with_mem[kind][idx], n):>13}"
            )


if __name__ == "__main__":
    main()
