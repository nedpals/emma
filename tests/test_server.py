import pytest
from unittest.mock import MagicMock, patch, AsyncMock


async def _mock_agent_run(events):
    """Create an async generator that yields the given events."""
    for event in events:
        yield event


@pytest.fixture
def mock_agent():
    agent = MagicMock()

    async def default_run(input, chat_history):
        yield {"type": "answer", "answer": "Test answer"}

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


def test_invoke_stream_contains_answer_event(client, mock_agent):
    async def fake_run(input, chat_history):
        yield {"type": "answer", "answer": "The answer is 42"}

    mock_agent.run = fake_run

    response = client.post("/invoke", json={
        "config": {},
        "input": {"input": "what is the answer?", "chat_history": [], "n_results": 10},
        "kwargs": {},
    })

    body = response.text
    assert "answer" in body
    assert "The answer is 42" in body


def test_invoke_stream_contains_tool_events(client, mock_agent):
    async def fake_run(input, chat_history):
        yield {"type": "tool_start", "tool": "search_handbook", "arguments": {"query": "test"}}
        yield {"type": "tool_end", "tool": "search_handbook", "success": True}
        yield {"type": "answer", "answer": "Found it"}

    mock_agent.run = fake_run

    response = client.post("/invoke", json={
        "config": {},
        "input": {"input": "search for something", "chat_history": [], "n_results": 10},
        "kwargs": {},
    })

    body = response.text
    assert "tool_start" in body
    assert "tool_end" in body
    assert "answer" in body
