import pytest


def test_tool_result_holds_fields():
    from tools import ToolResult
    r = ToolResult(content="found it", success=True)
    assert r.content == "found it"
    assert r.success is True


def test_tool_is_abstract():
    from tools import Tool
    with pytest.raises(TypeError):
        Tool()


def test_registry_register_and_get():
    from tools import Tool, ToolRegistry, ToolResult

    class FakeTool(Tool):
        name = "fake"
        description = "A fake tool"
        parameters = {"type": "object", "properties": {}}

        def execute(self, **kwargs) -> ToolResult:
            return ToolResult(content="ok", success=True)

    registry = ToolRegistry()
    tool = FakeTool()
    registry.register(tool)
    assert registry.get("fake") is tool


def test_registry_get_unknown_returns_none():
    from tools import ToolRegistry
    registry = ToolRegistry()
    assert registry.get("nonexistent") is None


def test_registry_get_tool_definitions():
    from tools import Tool, ToolRegistry, ToolResult

    class FakeTool(Tool):
        name = "fake"
        description = "A fake tool"
        parameters = {"type": "object", "properties": {"q": {"type": "string"}}}

        def execute(self, **kwargs) -> ToolResult:
            return ToolResult(content="ok", success=True)

    registry = ToolRegistry()
    registry.register(FakeTool())
    defs = registry.get_tool_definitions()

    assert len(defs) == 1
    assert defs[0].name == "fake"
    assert defs[0].description == "A fake tool"
    assert defs[0].parameters == {"type": "object", "properties": {"q": {"type": "string"}}}
