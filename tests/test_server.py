import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_agent():
    agent = MagicMock()

    async def default_run(input, chat_history):
        yield {"type": "answer_chunk", "chunk": "Test answer"}
        yield {"type": "answer_done"}

    agent.run = default_run
    return agent


@pytest.fixture
def client(mock_agent):
    from sse_starlette.sse import AppStatus
    AppStatus.should_exit_event = None
    AppStatus.should_exit = False

    with patch("agent_setup.create_agent", return_value=mock_agent):
        import importlib
        import server
        importlib.reload(server)
        from fastapi.testclient import TestClient
        yield TestClient(server.app)


def test_invoke_returns_sse_stream(client, mock_agent):
    response = client.post("/invoke", json={
        "config": {},
        "input": {"input": "test question", "chat_history": [], "n_results": 10},
        "kwargs": {},
    })

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


def test_invoke_stream_contains_answer_chunks(client, mock_agent):
    async def fake_run(input, chat_history):
        yield {"type": "answer_chunk", "chunk": "The answer"}
        yield {"type": "answer_chunk", "chunk": " is 42"}
        yield {"type": "answer_done"}

    mock_agent.run = fake_run

    response = client.post("/invoke", json={
        "config": {},
        "input": {"input": "what is the answer?", "chat_history": [], "n_results": 10},
        "kwargs": {},
    })

    body = response.text
    assert "answer_chunk" in body
    assert "The answer" in body
    assert "answer_done" in body


def test_invoke_stream_contains_tool_events(client, mock_agent):
    async def fake_run(input, chat_history):
        yield {"type": "tool_start", "tool": "search_handbook", "arguments": {"query": "test"}}
        yield {"type": "tool_end", "tool": "search_handbook", "success": True}
        yield {"type": "answer_chunk", "chunk": "Found it"}
        yield {"type": "answer_done"}

    mock_agent.run = fake_run

    response = client.post("/invoke", json={
        "config": {},
        "input": {"input": "search for something", "chat_history": [], "n_results": 10},
        "kwargs": {},
    })

    body = response.text
    assert "tool_start" in body
    assert "tool_end" in body
    assert "answer_chunk" in body
