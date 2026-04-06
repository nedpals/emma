from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, BinaryIO, Literal


@dataclass
class TextResponse:
    content: str


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallResponse:
    calls: list[ToolCall]


LLMResponse = TextResponse | ToolCallResponse


@dataclass
class ToolDefinition:
    """OpenAI function-calling schema format."""
    name: str
    description: str
    parameters: dict[str, Any]


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = -1) -> str:
        ...

    @abstractmethod
    def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolDefinition],
        temperature: float = 0.7,
    ) -> LLMResponse:
        ...

    @abstractmethod
    def embed(self, text: str, purpose: Literal["search_query", "search_document"]) -> list[float]:
        ...

    @abstractmethod
    def vision(
        self,
        image_data: str | BinaryIO | bytes,
        prompt: str,
        temperature: float = 0.1,
        response_format: Any | None = None,
    ) -> str | dict:
        ...
