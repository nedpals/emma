from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any, TypedDict, Union

from providers import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    SystemMessage,
    TextResponse,
    UserMessage,
)
from tools import ToolRegistry


class ToolStartEvent(TypedDict):
    type: str  # "tool_start"
    tool: str
    arguments: dict[str, Any]


class ToolEndEvent(TypedDict):
    type: str  # "tool_end"
    tool: str
    success: bool


class AnswerChunkEvent(TypedDict):
    type: str  # "answer_chunk"
    chunk: str


class AnswerDoneEvent(TypedDict):
    type: str  # "answer_done"


class ErrorEvent(TypedDict):
    type: str  # "error"
    message: str


AgentEvent = Union[ToolStartEvent, ToolEndEvent, AnswerChunkEvent, AnswerDoneEvent, ErrorEvent]


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        system_prompt: str,
        max_iterations: int = 5,
    ):
        self.provider = provider
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

    async def _stream_answer(self, messages: list[dict]) -> AsyncGenerator[AgentEvent, None]:
        """Stream the final text response token by token."""
        try:
            gen = self.provider.generate_stream(messages)
            while True:
                chunk = await asyncio.to_thread(next, gen, None)
                if chunk is None:
                    break
                yield AnswerChunkEvent(type="answer_chunk", chunk=chunk)
            yield AnswerDoneEvent(type="answer_done")
        except Exception as e:
            yield ErrorEvent(type="error", message=str(e))

    async def run(
        self,
        input: str,
        chat_history: list[ChatMessage],
    ) -> AsyncGenerator[AgentEvent, None]:
        messages: list[ChatMessage] = [
            SystemMessage(self.system_prompt),
            *chat_history,
            UserMessage(input),
        ]

        tool_defs = self.registry.get_tool_definitions()
        iterations = 0

        while iterations < self.max_iterations:
            # Force tool use on first iteration so the model always searches
            choice = "required" if iterations == 0 else "auto"
            try:
                response: LLMResponse = await asyncio.to_thread(
                    self.provider.generate_with_tools,
                    messages=messages,
                    tools=tool_defs,
                    tool_choice=choice,
                )
            except Exception as e:
                yield ErrorEvent(type="error", message=str(e))
                return

            if isinstance(response, TextResponse):
                # Re-run as streaming for the final answer
                async for event in self._stream_answer(messages):
                    yield event
                return

            for tc in response.calls:
                yield ToolStartEvent(type="tool_start", tool=tc.name, arguments=tc.arguments)

                tool = self.registry.get(tc.name)
                if tool is None:
                    tool_content = f"Error: Tool '{tc.name}' not found."
                    yield ToolEndEvent(type="tool_end", tool=tc.name, success=False)
                else:
                    try:
                        result = await asyncio.to_thread(tool.execute, **tc.arguments)
                        tool_content = result.content
                        yield ToolEndEvent(type="tool_end", tool=tc.name, success=result.success)
                    except Exception as e:
                        tool_content = f"Error: {e}"
                        yield ToolEndEvent(type="tool_end", tool=tc.name, success=False)

                messages.append(ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[{
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }],
                ))
                messages.append(ChatMessage(
                    role="tool",
                    tool_call_id=tc.id,
                    content=tool_content,
                ))

            iterations += 1

        # Soft cap reached
        messages.append(UserMessage("Please respond now with the information you have."))
        async for event in self._stream_answer(messages):
            yield event
