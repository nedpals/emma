from unittest.mock import MagicMock
import pytest

from providers import StreamDelta, ToolDefinition


def _mock_text_stream(text):
    """Create a generator that yields StreamDelta for text content."""
    def stream(*args, **kwargs):
        for word in text.split(" "):
            yield StreamDelta(content=word + " ")
        yield StreamDelta(finish_reason="stop")
    return stream


def _mock_tool_then_text_stream(tool_name, tool_args_json, text):
    """First call yields tool call deltas, second call yields text deltas."""
    calls = [0]
    def stream(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            yield StreamDelta(tool_call_id="call_1", tool_call_name=tool_name)
            yield StreamDelta(tool_call_arguments=tool_args_json)
            yield StreamDelta(finish_reason="tool_calls")
        else:
            for word in text.split(" "):
                yield StreamDelta(content=word + " ")
            yield StreamDelta(finish_reason="stop")
    return stream


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
        provider.generate_stream.side_effect = _mock_text_stream("Hello!")

    return Agent(
        provider=provider,
        registry=registry,
        system_prompt="You are a test assistant.",
        max_iterations=max_iterations,
    )


async def _collect_events(agent, input, chat_history=None):
    events = []
    async for event in agent.run(input, chat_history=chat_history or []):
        events.append(event)
    return events


def _get_answer(events):
    chunks = [e["chunk"] for e in events if e["type"] == "answer_chunk"]
    return "".join(chunks).strip() if chunks else None


@pytest.mark.asyncio
async def test_agent_streams_text_directly():
    agent = _make_agent()
    events = await _collect_events(agent, "hi")

    event_types = [e["type"] for e in events]
    assert "answer_chunk" in event_types
    assert "answer_done" in event_types
    assert _get_answer(events) == "Hello!"


@pytest.mark.asyncio
async def test_agent_executes_tool_then_streams_answer():
    provider = MagicMock()
    provider.generate_stream.side_effect = _mock_tool_then_text_stream(
        "echo", '{"text": "world"}', "The echo said: world"
    )

    agent = _make_agent(provider=provider)
    events = await _collect_events(agent, "echo world")

    assert _get_answer(events) == "The echo said: world"
    assert provider.generate_stream.call_count == 2


@pytest.mark.asyncio
async def test_agent_yields_tool_and_answer_events():
    provider = MagicMock()
    provider.generate_stream.side_effect = _mock_tool_then_text_stream(
        "echo", '{"text": "test"}', "done"
    )

    agent = _make_agent(provider=provider)
    events = await _collect_events(agent, "test")

    event_types = [e["type"] for e in events]
    assert "tool_start" in event_types
    assert "tool_end" in event_types
    assert "answer_chunk" in event_types
    assert "answer_done" in event_types


@pytest.mark.asyncio
async def test_agent_handles_unknown_tool():
    provider = MagicMock()
    provider.generate_stream.side_effect = _mock_tool_then_text_stream(
        "nonexistent", '{}', "Sorry"
    )

    agent = _make_agent(provider=provider)
    events = await _collect_events(agent, "use nonexistent")

    assert _get_answer(events) == "Sorry"


@pytest.mark.asyncio
async def test_agent_respects_max_iterations():
    call_count = [0]
    def always_tool_call(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:
            yield StreamDelta(tool_call_id="call_1", tool_call_name="echo")
            yield StreamDelta(tool_call_arguments='{"text": "loop"}')
            yield StreamDelta(finish_reason="tool_calls")
        else:
            for word in "Forced final".split(" "):
                yield StreamDelta(content=word + " ")
            yield StreamDelta(finish_reason="stop")

    provider = MagicMock()
    provider.generate_stream.side_effect = always_tool_call

    agent = _make_agent(provider=provider, max_iterations=2)
    events = await _collect_events(agent, "loop forever")

    assert _get_answer(events) == "Forced final"


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
    provider.generate_stream.side_effect = _mock_tool_then_text_stream(
        "fail", '{}', "Tool failed"
    )

    registry = ToolRegistry()
    registry.register(FailingTool())

    agent = Agent(
        provider=provider,
        registry=registry,
        system_prompt="Test",
        max_iterations=5,
    )

    events = await _collect_events(agent, "do something")
    assert _get_answer(events) == "Tool failed"


@pytest.mark.asyncio
async def test_agent_includes_system_prompt_in_messages():
    provider = MagicMock()
    provider.generate_stream.side_effect = _mock_text_stream("hi")

    agent = _make_agent(provider=provider)
    await _collect_events(agent, "hello")

    call_args = provider.generate_stream.call_args
    messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
    assert messages[0].role == "system"
    assert "test assistant" in messages[0].content


@pytest.mark.asyncio
async def test_agent_yields_error_on_provider_failure():
    provider = MagicMock()
    def failing_stream(*args, **kwargs):
        raise RuntimeError("LM Studio is down")
    provider.generate_stream.side_effect = failing_stream

    agent = _make_agent(provider=provider)
    events = await _collect_events(agent, "hello")

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "LM Studio is down" in events[0]["message"]
