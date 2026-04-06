from __future__ import annotations

from typing import Any, Callable

from providers import (
    LLMProvider,
    LLMResponse,
    TextResponse,
)
from tools import ToolRegistry


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

    def run(
        self,
        input: str,
        chat_history: list[dict],
        on_event: Callable[[dict], Any] | None = None,
    ) -> str:
        def emit(event: dict) -> None:
            if on_event is not None:
                on_event(event)

        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            *chat_history,
            {"role": "user", "content": input},
        ]

        tool_defs = self.registry.get_tool_definitions()
        iterations = 0

        while iterations < self.max_iterations:
            response: LLMResponse = self.provider.generate_with_tools(
                messages=messages,
                tools=tool_defs,
            )

            if isinstance(response, TextResponse):
                emit({"type": "answer", "answer": response.content})
                return response.content

            for tc in response.calls:
                emit({"type": "tool_start", "tool": tc.name, "arguments": tc.arguments})

                tool = self.registry.get(tc.name)
                if tool is None:
                    tool_content = f"Error: Tool '{tc.name}' not found."
                    emit({"type": "tool_end", "tool": tc.name, "success": False})
                else:
                    try:
                        result = tool.execute(**tc.arguments)
                        tool_content = result.content
                        emit({"type": "tool_end", "tool": tc.name, "success": result.success})
                    except Exception as e:
                        tool_content = f"Error: {e}"
                        emit({"type": "tool_end", "tool": tc.name, "success": False})

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": str(tc.arguments)},
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
        final = self.provider.generate(messages)
        emit({"type": "answer", "answer": final})
        return final
