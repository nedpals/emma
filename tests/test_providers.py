import pytest


def test_text_response_holds_content():
    from providers import TextResponse
    r = TextResponse(content="hello")
    assert r.content == "hello"


def test_tool_call_holds_fields():
    from providers import ToolCall
    tc = ToolCall(id="1", name="search", arguments={"query": "test"})
    assert tc.id == "1"
    assert tc.name == "search"
    assert tc.arguments == {"query": "test"}


def test_tool_call_response_holds_calls():
    from providers import ToolCall, ToolCallResponse
    tc = ToolCall(id="1", name="search", arguments={})
    r = ToolCallResponse(calls=[tc])
    assert len(r.calls) == 1
    assert r.calls[0].name == "search"


def test_llm_provider_is_abstract():
    from providers import LLMProvider
    with pytest.raises(TypeError):
        LLMProvider()


def test_llm_provider_declares_required_methods():
    from providers import LLMProvider
    assert hasattr(LLMProvider, "generate")
    assert hasattr(LLMProvider, "generate_with_tools")
    assert hasattr(LLMProvider, "embed")
    assert hasattr(LLMProvider, "vision")


from unittest.mock import MagicMock, patch


def test_lm_studio_provider_is_concrete():
    from providers.lm_studio import LMStudioProvider
    provider = LMStudioProvider(
        base_url="http://localhost:1234/v1",
        api_key="test",
        llm_model="test-model",
        vlm_model="test-vlm",
        embedding_model="test-embed",
    )
    assert provider is not None


def test_lm_studio_generate_calls_openai():
    from providers.lm_studio import LMStudioProvider
    provider = LMStudioProvider(
        base_url="http://localhost:1234/v1",
        api_key="test",
        llm_model="test-model",
        vlm_model="test-vlm",
        embedding_model="test-embed",
    )

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "  test response  "

    with patch.object(provider._client.chat.completions, "create", return_value=mock_response) as mock_create:
        from providers import UserMessage
        result = provider.generate(
            messages=[UserMessage("hi")],
            temperature=0.5,
        )

    assert result == "test response"
    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args[1]
    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["temperature"] == 0.5


def test_lm_studio_generate_with_tools_returns_text():
    from providers import TextResponse, ToolDefinition
    from providers.lm_studio import LMStudioProvider
    provider = LMStudioProvider(
        base_url="http://localhost:1234/v1",
        api_key="test",
        llm_model="test-model",
        vlm_model="test-vlm",
        embedding_model="test-embed",
    )

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = "just text"

    with patch.object(provider._client.chat.completions, "create", return_value=mock_response):
        from providers import UserMessage
        result = provider.generate_with_tools(
            messages=[UserMessage("hi")],
            tools=[ToolDefinition(name="test", description="test tool", parameters={"type": "object", "properties": {}})],
        )

    assert isinstance(result, TextResponse)
    assert result.content == "just text"


def test_lm_studio_generate_with_tools_returns_tool_calls():
    from providers import ToolCallResponse, ToolDefinition
    from providers.lm_studio import LMStudioProvider
    provider = LMStudioProvider(
        base_url="http://localhost:1234/v1",
        api_key="test",
        llm_model="test-model",
        vlm_model="test-vlm",
        embedding_model="test-embed",
    )

    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function.name = "search_handbook"
    mock_tool_call.function.arguments = '{"query": "attendance"}'

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = [mock_tool_call]
    mock_response.choices[0].message.content = None

    with patch.object(provider._client.chat.completions, "create", return_value=mock_response):
        from providers import UserMessage
        result = provider.generate_with_tools(
            messages=[UserMessage("attendance policy")],
            tools=[ToolDefinition(name="search_handbook", description="search", parameters={"type": "object", "properties": {}})],
        )

    assert isinstance(result, ToolCallResponse)
    assert len(result.calls) == 1
    assert result.calls[0].name == "search_handbook"
    assert result.calls[0].arguments == {"query": "attendance"}


def test_lm_studio_embed():
    from providers.lm_studio import LMStudioProvider
    provider = LMStudioProvider(
        base_url="http://localhost:1234/v1",
        api_key="test",
        llm_model="test-model",
        vlm_model="test-vlm",
        embedding_model="test-embed",
    )

    mock_response = MagicMock()
    mock_response.data = [MagicMock()]
    mock_response.data[0].embedding = [0.1, 0.2, 0.3]

    with patch.object(provider._client.embeddings, "create", return_value=mock_response) as mock_create:
        result = provider.embed("test text", "search_query")

    assert result == [0.1, 0.2, 0.3]
    call_kwargs = mock_create.call_args[1]
    assert call_kwargs["input"] == ["search_query: test text"]
    assert call_kwargs["model"] == "test-embed"


def test_stream_delta_holds_content():
    from providers import StreamDelta
    d = StreamDelta(content="hello")
    assert d.content == "hello"
    assert d.finish_reason is None


def test_stream_delta_holds_tool_call():
    from providers import StreamDelta
    d = StreamDelta(tool_call_id="1", tool_call_name="search", tool_call_arguments='{"q": "test"}')
    assert d.tool_call_name == "search"
    assert d.content is None


def test_stream_delta_holds_finish_reason():
    from providers import StreamDelta
    d = StreamDelta(finish_reason="stop")
    assert d.finish_reason == "stop"


def test_lm_studio_stream_yields_content_deltas():
    from providers import StreamDelta, UserMessage
    from providers.lm_studio import LMStudioProvider
    provider = LMStudioProvider(
        base_url="http://localhost:1234/v1",
        api_key="test",
        llm_model="test-model",
        vlm_model="test-vlm",
        embedding_model="test-embed",
    )

    mock_chunk_1 = MagicMock()
    mock_chunk_1.choices = [MagicMock()]
    mock_chunk_1.choices[0].delta.content = "Hello "
    mock_chunk_1.choices[0].delta.tool_calls = None
    mock_chunk_1.choices[0].finish_reason = None

    mock_chunk_2 = MagicMock()
    mock_chunk_2.choices = [MagicMock()]
    mock_chunk_2.choices[0].delta.content = "world"
    mock_chunk_2.choices[0].delta.tool_calls = None
    mock_chunk_2.choices[0].finish_reason = None

    mock_chunk_3 = MagicMock()
    mock_chunk_3.choices = [MagicMock()]
    mock_chunk_3.choices[0].delta.content = None
    mock_chunk_3.choices[0].delta.tool_calls = None
    mock_chunk_3.choices[0].finish_reason = "stop"

    with patch.object(provider._client.chat.completions, "create", return_value=iter([mock_chunk_1, mock_chunk_2, mock_chunk_3])):
        deltas = list(provider.generate_stream(messages=[UserMessage("hi")]))

    contents = [d.content for d in deltas if d.content]
    assert contents == ["Hello ", "world"]
    assert deltas[-1].finish_reason == "stop"


def test_lm_studio_stream_yields_tool_call_deltas():
    from providers import StreamDelta, ToolDefinition, UserMessage
    from providers.lm_studio import LMStudioProvider
    provider = LMStudioProvider(
        base_url="http://localhost:1234/v1",
        api_key="test",
        llm_model="test-model",
        vlm_model="test-vlm",
        embedding_model="test-embed",
    )

    mock_tc = MagicMock()
    mock_tc.index = 0
    mock_tc.id = "call_123"
    mock_tc.function.name = "search_handbook"
    mock_tc.function.arguments = '{"query":'

    mock_tc2 = MagicMock()
    mock_tc2.index = 0
    mock_tc2.id = None
    mock_tc2.function.name = None
    mock_tc2.function.arguments = ' "attendance"}'

    mock_chunk_1 = MagicMock()
    mock_chunk_1.choices = [MagicMock()]
    mock_chunk_1.choices[0].delta.content = None
    mock_chunk_1.choices[0].delta.tool_calls = [mock_tc]
    mock_chunk_1.choices[0].finish_reason = None

    mock_chunk_2 = MagicMock()
    mock_chunk_2.choices = [MagicMock()]
    mock_chunk_2.choices[0].delta.content = None
    mock_chunk_2.choices[0].delta.tool_calls = [mock_tc2]
    mock_chunk_2.choices[0].finish_reason = None

    mock_chunk_3 = MagicMock()
    mock_chunk_3.choices = [MagicMock()]
    mock_chunk_3.choices[0].delta.content = None
    mock_chunk_3.choices[0].delta.tool_calls = None
    mock_chunk_3.choices[0].finish_reason = "tool_calls"

    tools = [ToolDefinition(name="search_handbook", description="search", parameters={"type": "object", "properties": {}})]

    with patch.object(provider._client.chat.completions, "create", return_value=iter([mock_chunk_1, mock_chunk_2, mock_chunk_3])):
        deltas = list(provider.generate_stream(messages=[UserMessage("test")], tools=tools))

    tool_deltas = [d for d in deltas if d.tool_call_id or d.tool_call_arguments]
    assert len(tool_deltas) >= 1
    assert deltas[-1].finish_reason == "tool_calls"
