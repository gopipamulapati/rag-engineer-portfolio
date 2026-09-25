import sqlite3

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from app.graph import ask, build_graph, session_config, turn_input
from app.reasoning import NOT_FOUND


def test_follow_up_is_resolved_from_memory(retriever):
    graph = build_graph(retriever)
    ask(graph, "s1", "How often are ClickHouse backups taken?")
    result = ask(graph, "s1", "How long are they retained?")
    assert "clickhouse" in result["standalone_query"].lower()
    assert "14 days" in result["answer"]


def test_same_follow_up_without_memory_lacks_context(retriever):
    graph = build_graph(retriever)
    result = ask(graph, "fresh", "How long are they retained?")
    assert result["standalone_query"] == "How long are they retained?"
    assert "14 days" not in result["answer"]


def test_sessions_are_isolated(retriever):
    graph = build_graph(retriever)
    ask(graph, "alice", "How often are ClickHouse backups taken?")
    ask(graph, "bob", "How is regional failover triggered?")
    alice = ask(graph, "alice", "How long are they retained?")
    assert "failover" not in alice["standalone_query"].lower()
    messages = graph.get_state(session_config("alice")).values["messages"]
    assert len(messages) == 4  # two user turns + two answers


def test_out_of_domain_question_is_declined(retriever):
    result = ask(build_graph(retriever), "s", "How do I bake a chocolate cake?")
    assert result["answer"] == NOT_FOUND
    assert result["sources"] == []


def test_sqlite_memory_survives_a_restart(retriever, tmp_path):
    db = tmp_path / "sessions.db"
    first = build_graph(
        retriever, checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False))
    )
    ask(first, "s1", "How often are ClickHouse backups taken?")

    # A brand-new graph and connection, like a restarted server.
    second = build_graph(
        retriever, checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False))
    )
    result = ask(second, "s1", "How long are they retained?")
    assert "14 days" in result["answer"]


def test_llm_condense_rewrites_the_follow_up(retriever):
    llm = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage("Backups are taken every 6 hours [data_retention_policy.txt]"),
                AIMessage("How long are ClickHouse backups retained?"),  # condense output
                AIMessage(
                    "ClickHouse backups are retained for 14 days [data_retention_policy.txt]"
                ),
            ]
        )
    )
    graph = build_graph(retriever, llm=llm)
    ask(graph, "s1", "How often are ClickHouse backups taken?")
    result = ask(graph, "s1", "And how long are they kept?")
    assert result["standalone_query"] == "How long are ClickHouse backups retained?"
    assert result["sources"][0] == "data_retention_policy.txt"
    assert "14 days" in result["answer"]


def test_ungrounded_llm_answer_is_retried_then_declined(retriever):
    llm = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage("Backups are stored on magnetic tape in Antarctica forever."),
                AIMessage("Backups are stored on magnetic tape in Antarctica forever."),
            ]
        )
    )
    result = ask(build_graph(retriever, llm=llm), "s1", "How long are ClickHouse backups retained?")
    assert result["retrieval_attempts"] == 2
    assert result["answer"] == NOT_FOUND


def test_llm_answer_tokens_stream_from_the_generate_node(retriever):
    llm = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    "Failover is triggered manually by the incident commander "
                    "[incident_response_runbook.txt]"
                ),
            ]
        )
    )
    graph = build_graph(retriever, llm=llm)
    tokens = [
        chunk.content
        for chunk, meta in graph.stream(
            turn_input("How is regional failover triggered?"),
            session_config("s1"),
            stream_mode="messages",
        )
        if meta["langgraph_node"] == "generate" and chunk.content
    ]
    assert len(tokens) > 1  # arrived in pieces, not as one message
    assert "manually" in "".join(tokens)
