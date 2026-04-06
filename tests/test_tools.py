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


def test_calculate_simple_expression():
    from tools.calculate import CalculateTool
    tool = CalculateTool()
    result = tool.execute(expression="3 * 8")
    assert result.success is True
    assert result.content == "24"


def test_calculate_float_result():
    from tools.calculate import CalculateTool
    tool = CalculateTool()
    result = tool.execute(expression="10 / 3")
    assert result.success is True
    assert "3.333" in result.content


def test_calculate_rejects_dangerous_input():
    from tools.calculate import CalculateTool
    tool = CalculateTool()
    result = tool.execute(expression="__import__('os').system('ls')")
    assert result.success is False
    assert "Error" in result.content


def test_calculate_rejects_non_math():
    from tools.calculate import CalculateTool
    tool = CalculateTool()
    result = tool.execute(expression="open('/etc/passwd')")
    assert result.success is False


def test_calculate_tool_metadata():
    from tools.calculate import CalculateTool
    tool = CalculateTool()
    assert tool.name == "calculate"
    assert "math" in tool.description.lower() or "calcul" in tool.description.lower()


import json
import os
import tempfile


def test_get_page_returns_segments_for_page():
    from tools.get_page import GetPageTool

    with tempfile.TemporaryDirectory() as tmpdir:
        data = {
            "segments": [
                {"page_number": "5", "context": "Attendance", "text_segment": "3 tardies equal 1 absence"},
                {"page_number": "5", "context": "Attendance", "text_segment": "Max 6 absences per semester"},
                {"page_number": "6", "context": "Grading", "text_segment": "Grading scale info"},
            ]
        }
        path = os.path.join(tmpdir, "page_0.json")
        with open(path, "w") as f:
            json.dump(data, f)

        tool = GetPageTool(extracted_dir=tmpdir)
        result = tool.execute(page_number=5)

    assert result.success is True
    assert "3 tardies equal 1 absence" in result.content
    assert "Max 6 absences per semester" in result.content
    assert "Grading scale info" not in result.content


def test_get_page_not_found():
    from tools.get_page import GetPageTool

    with tempfile.TemporaryDirectory() as tmpdir:
        data = {"segments": [{"page_number": "1", "context": "Intro", "text_segment": "Hello"}]}
        path = os.path.join(tmpdir, "page_0.json")
        with open(path, "w") as f:
            json.dump(data, f)

        tool = GetPageTool(extracted_dir=tmpdir)
        result = tool.execute(page_number=999)

    assert result.success is False
    assert "No content found" in result.content


def test_get_page_tool_metadata():
    from tools.get_page import GetPageTool
    tool = GetPageTool(extracted_dir="/tmp")
    assert tool.name == "get_page"


from unittest.mock import MagicMock, patch


def test_search_handbook_returns_results_with_grounding_reminder():
    from tools.search_handbook import SearchHandbookTool

    mock_provider = MagicMock()
    mock_provider.embed.return_value = [0.1, 0.2, 0.3]

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["Doc about attendance policy", "Doc about grading"]],
    }

    tool = SearchHandbookTool(provider=mock_provider, collection=mock_collection)

    with patch("tools.search_handbook.extract_keywords", return_value=["attendance"]):
        result = tool.execute(query="attendance policy")

    assert result.success is True
    assert "Doc about attendance policy" in result.content
    assert "Answer using ONLY the information above" in result.content
    mock_provider.embed.assert_called_once_with("attendance policy", "search_query")


def test_search_handbook_no_results():
    from tools.search_handbook import SearchHandbookTool

    mock_provider = MagicMock()
    mock_provider.embed.return_value = [0.1, 0.2, 0.3]

    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": [[]]}

    tool = SearchHandbookTool(provider=mock_provider, collection=mock_collection)

    with patch("tools.search_handbook.extract_keywords", return_value=[]):
        result = tool.execute(query="nonexistent topic")

    assert result.success is True
    assert "No relevant information found" in result.content


def test_search_handbook_tool_metadata():
    from tools.search_handbook import SearchHandbookTool
    tool = SearchHandbookTool(provider=MagicMock(), collection=MagicMock())
    assert tool.name == "search_handbook"
    assert "query" in tool.parameters["properties"]
