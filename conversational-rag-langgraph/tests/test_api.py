from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["mode"] == "heuristic"


def test_multi_turn_chat_history_and_delete():
    sid = "api-test"
    client.post(
        "/chat", json={"session_id": sid, "message": "How often are ClickHouse backups taken?"}
    )
    r = client.post("/chat", json={"session_id": sid, "message": "How long are they retained?"})
    assert r.status_code == 200
    assert "14 days" in r.json()["answer"]

    history = client.get(f"/sessions/{sid}/history").json()["messages"]
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]

    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert client.get(f"/sessions/{sid}/history").status_code == 404


def test_stream_emits_steps_tokens_and_answer():
    with client.stream(
        "POST",
        "/chat/stream",
        json={"session_id": "stream", "message": "How is regional failover triggered?"},
    ) as r:
        body = "".join(r.iter_text())
    assert "event: step" in body and '"node": "retrieve"' in body
    assert "event: token" in body
    assert "event: answer" in body and "manually" in body


def test_validation():
    assert client.post("/chat", json={"session_id": "", "message": "hi"}).status_code == 422
