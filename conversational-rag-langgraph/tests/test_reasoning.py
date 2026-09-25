from app import reasoning
from app.retrieval import Chunk, Hit


def test_follow_up_detection():
    assert reasoning.is_follow_up("How long are they kept?")
    assert reasoning.is_follow_up("What about hour-level rollups?")
    assert not reasoning.is_follow_up("How is regional failover triggered?")


def test_condense_carries_topic_into_follow_up():
    _, topic = reasoning.heuristic_condense("How often are ClickHouse backups taken?", None)
    query, _ = reasoning.heuristic_condense("How long are they retained?", topic)
    assert "clickhouse" in query and "backups" in query


def test_condense_leaves_standalone_questions_alone():
    query, topic = reasoning.heuristic_condense(
        "How is regional failover triggered?", ["clickhouse", "backups"]
    )
    assert query == "How is regional failover triggered?"
    assert "clickhouse" not in topic


def test_topic_is_capped_so_old_context_fades():
    topic = None
    for q in [
        "How is regional failover triggered?",
        "Why is it manual?",
        "What does it update?",
        "How long does it take?",
    ]:
        _, topic = reasoning.heuristic_condense(q, topic)
    assert len(topic) <= reasoning.TOPIC_SIZE


def test_heuristic_answer_cites_source():
    hits = [Hit(Chunk("doc.txt", "Backups are taken every 6 hours. They are encrypted."), 1.0)]
    answer = reasoning.heuristic_answer("How often are backups taken?", "backups taken", hits)
    assert "6 hours" in answer and "[doc.txt]" in answer


def test_grounding_score():
    hits = [Hit(Chunk("doc.txt", "Backups are taken every 6 hours."), 1.0)]
    assert reasoning.grounding_score("Backups are taken every 6 hours [doc.txt]", hits) == 1.0
    assert reasoning.grounding_score("Backups are stored on the moon.", hits) < 0.6
    assert reasoning.grounding_score(reasoning.NOT_FOUND, []) == 1.0
    # off-topic retrieval: none of the question's words are in the context
    assert reasoning.grounding_score("Backups are taken.", hits, "How do I bake a cake?") == 0.0
