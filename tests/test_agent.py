from unittest.mock import MagicMock
import pytest

from providers import TextResponse, ToolCall, ToolCallResponse


def _make_agent(provider=None, max_iterations=5):
    from agent import Agent
    from tools import Tool, ToolRegistry, ToolResult

    class EchoTool(Tool):
        name = "echo"
        description = "Echoes input"
        parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

        def execute(self, **kwargs) -> ToolResult:
            return ToolResult(content=f"Echo: {kwargs['text']}", success=True)

    registry = ToolRegistry()
    registry.register(EchoTool())

    if provider is None:
        provider = MagicMock()
        provider.generate_with_tools.return_value = TextResponse(content="Hello!")
        provider.generate.return_value = "Forced response"

    return Agent(
        provider=provider,
        registry=registry,
        system_prompt="You are a test assistant.",
        max_iterations=max_iterations,
    )


async def _collect_events(agent, input, chat_history=None):
    """Helper to collect all events from the async generator."""
    events = []
    async for event in agent.run(input, chat_history=chat_history or []):
        events.append(event)
    return events


def _get_answer(events):
    """Extract the answer from collected events."""
    for event in events:
        if event["type"] == "answer":
            return event["answer"]
    return None


@pytest.mark.asyncio
async def test_agent_returns_text_response_directly():
    agent = _make_agent()
    events = await _collect_events(agent, "hi")
    assert _get_answer(events) == "Hello!"
    assert len(events) == 1


@pytest.mark.asyncio
async def test_agent_executes_tool_and_returns_final_response():
    provider = MagicMock()
    provider.generate_with_tools.side_effect = [
        ToolCallResponse(calls=[ToolCall(id="1", name="echo", arguments={"text": "world"})]),
        TextResponse(content="The echo said: world"),
    ]

    agent = _make_agent(provider=provider)
    events = await _collect_events(agent, "echo world")

    assert _get_answer(events) == "The echo said: world"
    assert provider.generate_with_tools.call_count == 2


@pytest.mark.asyncio
async def test_agent_yields_tool_events():
    provider = MagicMock()
    provider.generate_with_tools.side_effect = [
        ToolCallResponse(calls=[ToolCall(id="1", name="echo", arguments={"text": "test"})]),
        TextResponse(content="done"),
    ]

    agent = _make_agent(provider=provider)
    events = await _collect_events(agent, "test")

    event_types = [e["type"] for e in events]
    assert "tool_start" in event_types
    assert "tool_end" in event_types
    assert "answer" in event_types


@pytest.mark.asyncio
async def test_agent_handles_unknown_tool():
    provider = MagicMock()
    provider.generate_with_tools.side_effect = [
        ToolCallResponse(calls=[ToolCall(id="1", name="nonexistent", arguments={})]),
        TextResponse(content="Sorry, I couldn't find that tool"),
    ]

    agent = _make_agent(provider=provider)
    events = await _collect_events(agent, "use nonexistent")

    assert _get_answer(events) == "Sorry, I couldn't find that tool"


@pytest.mark.asyncio
async def test_agent_respects_max_iterations():
    provider = MagicMock()
    provider.generate_with_tools.return_value = ToolCallResponse(
        calls=[ToolCall(id="1", name="echo", arguments={"text": "loop"})]
    )
    provider.generate.return_value = "Forced final response"

    agent = _make_agent(provider=provider, max_iterations=2)
    events = await _collect_events(agent, "loop forever")

    assert _get_answer(events) == "Forced final response"
    assert provider.generate_with_tools.call_count == 2
    assert provider.generate.call_count == 1


@pytest.mark.asyncio
async def test_agent_handles_tool_execution_error():
    from agent import Agent
    from tools import Tool, ToolRegistry, ToolResult

    class FailingTool(Tool):
        name = "fail"
        description = "Always fails"
        parameters = {"type": "object", "properties": {}}

        def execute(self, **kwargs) -> ToolResult:
            raise RuntimeError("Something broke")

    provider = MagicMock()
    provider.generate_with_tools.side_effect = [
        ToolCallResponse(calls=[ToolCall(id="1", name="fail", arguments={})]),
        TextResponse(content="Tool failed, here's my best answer"),
    ]

    registry = ToolRegistry()
    registry.register(FailingTool())

    agent = Agent(
        provider=provider,
        registry=registry,
        system_prompt="Test",
        max_iterations=5,
    )

    events = await _collect_events(agent, "do something")
    assert _get_answer(events) == "Tool failed, here's my best answer"


@pytest.mark.asyncio
async def test_agent_includes_system_prompt_in_messages():
    provider = MagicMock()
    provider.generate_with_tools.return_value = TextResponse(content="hi")

    agent = _make_agent(provider=provider)
    await _collect_events(agent, "hello")

    call_args = provider.generate_with_tools.call_args
    messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
    assert messages[0]["role"] == "system"
    assert "test assistant" in messages[0]["content"]
