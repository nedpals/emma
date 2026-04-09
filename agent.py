from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any, TypedDict, Union

from providers import (
    ChatMessage,
    LLMProvider,
    StreamDelta,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from tools import ToolRegistry


class ToolStartEvent(TypedDict):
    type: str
    tool: str
    arguments: dict[str, Any]


class ToolEndEvent(TypedDict):
    type: str
    tool: str
    success: bool


class AnswerChunkEvent(TypedDict):
    type: str
    chunk: str


class AnswerDoneEvent(TypedDict):
    type: str


class ErrorEvent(TypedDict):
    type: str
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

    def _parse_tool_calls(self, deltas: list[StreamDelta]) -> list[ToolCall]:
        """Parse buffered tool call deltas into ToolCall objects."""
        calls: dict[int, dict] = {}
        idx = 0
        for d in deltas:
            if d.tool_call_id:
                calls[idx] = {"id": d.tool_call_id, "name": "", "arguments": ""}
            if d.tool_call_name:
                calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                calls[idx]["name"] = d.tool_call_name
            if d.tool_call_arguments:
                calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                calls[idx]["arguments"] += d.tool_call_arguments
            if d.tool_call_id and len(calls) > 1:
                idx += 1

        result = []
        for tc_data in calls.values():
            try:
                args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            result.append(ToolCall(id=tc_data["id"], name=tc_data["name"], arguments=args))
        return result

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
            choice = "auto"
            tool_call_buffer: list[StreamDelta] = []
            finish_reason = None
            thinking = False
            content_buffer = ""

            try:
                gen = self.provider.generate_stream(
                    messages=messages,
                    tools=tool_defs,
                    tool_choice=choice,
                )
                while True:
                    delta = await asyncio.to_thread(next, gen, None)
                    if delta is None:
                        break

                    if delta.content:
                        content_buffer += delta.content

                        # Detect thinking block boundaries
                        while True:
                            if thinking:
                                end = content_buffer.find("<channel|>")
                                if end == -1:
                                    content_buffer = ""
                                    break
                                content_buffer = content_buffer[end + len("<channel|>"):]
                                thinking = False
                            else:
                                start = content_buffer.find("<|channel>")
                                if start == -1:
                                    # No thinking tag — flush buffer as answer
                                    if content_buffer:
                                        yield AnswerChunkEvent(type="answer_chunk", chunk=content_buffer)
                                        content_buffer = ""
                                    break
                                # Flush content before the thinking tag
                                if start > 0:
                                    yield AnswerChunkEvent(type="answer_chunk", chunk=content_buffer[:start])
                                content_buffer = content_buffer[start + len("<|channel>"):]
                                thinking = True

                    if delta.tool_call_id or delta.tool_call_name or delta.tool_call_arguments:
                        tool_call_buffer.append(delta)

                    if delta.finish_reason:
                        finish_reason = delta.finish_reason
            except Exception as e:
                yield ErrorEvent(type="error", message=str(e))
                return

            if finish_reason == "stop":
                yield AnswerDoneEvent(type="answer_done")
                return

            if finish_reason == "tool_calls" and tool_call_buffer:
                parsed_calls = self._parse_tool_calls(tool_call_buffer)

                for tc in parsed_calls:
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

        # Soft cap — stream without tools
        messages.append(UserMessage("Please respond now with the information you have."))
        thinking = False
        content_buffer = ""
        try:
            gen = self.provider.generate_stream(messages=messages)
            while True:
                delta = await asyncio.to_thread(next, gen, None)
                if delta is None:
                    break
                if delta.content:
                    content_buffer += delta.content
                    while True:
                        if thinking:
                            end = content_buffer.find("<channel|>")
                            if end == -1:
                                content_buffer = ""
                                break
                            content_buffer = content_buffer[end + len("<channel|>"):]
                            thinking = False
                        else:
                            start = content_buffer.find("<|channel>")
                            if start == -1:
                                if content_buffer:
                                    yield AnswerChunkEvent(type="answer_chunk", chunk=content_buffer)
                                    content_buffer = ""
                                break
                            if start > 0:
                                yield AnswerChunkEvent(type="answer_chunk", chunk=content_buffer[:start])
                            content_buffer = content_buffer[start + len("<|channel>"):]
                            thinking = True
                if delta.finish_reason:
                    yield AnswerDoneEvent(type="answer_done")
                    return
        except Exception as e:
            yield ErrorEvent(type="error", message=str(e))
