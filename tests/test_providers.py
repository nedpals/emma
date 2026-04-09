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
