import json
import os

from tools import Tool, ToolResult


class GetPageTool(Tool):
    name = "get_page"
    description = "Retrieve the full extracted content for a specific handbook page number."
    parameters = {
        "type": "object",
        "properties": {
            "page_number": {
                "type": "integer",
                "description": "The handbook page number to look up",
            },
        },
        "required": ["page_number"],
    }

    def __init__(self, extracted_dir: str = "./extracted_2"):
        self._extracted_dir = extracted_dir
        self._segments: list[dict] | None = None

    def _load_segments(self) -> list[dict]:
        if self._segments is None:
            segments = []
            for filename in sorted(os.listdir(self._extracted_dir)):
                if not filename.endswith(".json"):
                    continue
                filepath = os.path.join(self._extracted_dir, filename)
                with open(filepath) as f:
                    data = json.load(f)
                segments.extend(data.get("segments", []))
            self._segments = segments
        return self._segments

    def execute(self, **kwargs) -> ToolResult:
        page_number = kwargs.get("page_number")
        if page_number is None:
            return ToolResult(content="Error: page_number is required", success=False)

        page_str = str(page_number)
        segments = self._load_segments()
        matches = [s for s in segments if s.get("page_number") == page_str]

        if not matches:
            return ToolResult(
                content=f"No content found for page {page_number}.",
                success=False,
            )

        parts = []
        for seg in matches:
            context = seg.get("context", "")
            text = seg.get("text_segment", "")
            parts.append(f"[{context}]\n{text}")

        content = f"## Handbook Page {page_number}\n\n" + "\n\n".join(parts)
        return ToolResult(content=content, success=True)
