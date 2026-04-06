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
