from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, BinaryIO, Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None



class SystemMessage(ChatMessage):
    def __init__(self, content: str, **kwargs: Any):
        super().__init__(role="system", content=content, **kwargs)


class UserMessage(ChatMessage):
    def __init__(self, content: str, **kwargs: Any):
        super().__init__(role="user", content=content, **kwargs)


class AIMessage(ChatMessage):
    def __init__(self, content: str, **kwargs: Any):
        super().__init__(role="assistant", content=content, **kwargs)


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
    def generate(self, messages: list[ChatMessage], temperature: float = 0.7, max_tokens: int = -1) -> str:
        ...

    @abstractmethod
    def generate_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        temperature: float = 0.7,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        ...

    @abstractmethod
    def generate_stream(self, messages: list[ChatMessage], temperature: float = 0.7) -> Generator[str, None, None]:
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
