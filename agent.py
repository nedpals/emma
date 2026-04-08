from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any, TypedDict, Union

from providers import (
    LLMProvider,
    LLMResponse,
    TextResponse,
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


class AnswerEvent(TypedDict):
    type: str  # "answer"
    answer: str


class ErrorEvent(TypedDict):
    type: str  # "error"
    message: str


AgentEvent = Union[ToolStartEvent, ToolEndEvent, AnswerEvent, ErrorEvent]


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

    async def run(
        self,
        input: str,
        chat_history: list[dict],
    ) -> AsyncGenerator[AgentEvent, None]:
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            *chat_history,
            {"role": "user", "content": input},
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
                yield AnswerEvent(type="answer", answer=response.content)
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

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_content,
                })

            iterations += 1

        # Soft cap reached
        messages.append({
            "role": "user",
            "content": "Please respond now with the information you have.",
        })
        try:
            final = await asyncio.to_thread(self.provider.generate, messages)
            yield AnswerEvent(type="answer", answer=final)
        except Exception as e:
            yield ErrorEvent(type="error", message=str(e))
