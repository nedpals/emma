from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any, BinaryIO, Literal

import lmstudio as lms
from openai import OpenAI

from providers import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    TextResponse,
    ToolCall,
    ToolCallResponse,
    ToolDefinition,
)


def _to_dicts(messages: list[ChatMessage]) -> list[dict]:
    return [m.model_dump(exclude_none=True) for m in messages]


class LMStudioProvider(LLMProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        llm_model: str,
        vlm_model: str,
        embedding_model: str,
    ):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._llm_model = llm_model
        self._vlm_model = vlm_model
        self._embedding_model = embedding_model

    def generate(self, messages: list[ChatMessage], temperature: float = 0.7, max_tokens: int = -1) -> str:
        params: dict[str, Any] = {
            "model": self._llm_model,
            "messages": _to_dicts(messages),
            "temperature": temperature,
        }
        if max_tokens > 0:
            params["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()

    def generate_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        temperature: float = 0.7,
        tool_choice: str = "auto",
    ) -> LLMResponse:
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

        response = self._client.chat.completions.create(
            model=self._llm_model,
            messages=_to_dicts(messages),
            tools=openai_tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )

        message = response.choices[0].message

        if message.tool_calls:
            calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in message.tool_calls
            ]
            return ToolCallResponse(calls=calls)

        return TextResponse(content=message.content.strip() if message.content else "")

    def generate_stream(self, messages: list[ChatMessage], temperature: float = 0.7) -> Generator[str, None, None]:
        stream = self._client.chat.completions.create(
            model=self._llm_model,
            messages=_to_dicts(messages),
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def embed(self, text: str, purpose: Literal["search_query", "search_document"]) -> list[float]:
        response = self._client.embeddings.create(
            model=self._embedding_model,
            input=[f"{purpose}: {text}"],
        )
        return response.data[0].embedding

    def vision(
        self,
        image_data: str | BinaryIO | bytes,
        prompt: str,
        temperature: float = 0.1,
        response_format: Any | None = None,
    ) -> str | dict:
        with lms.Client() as client:
            image_handle = client.files.prepare_image(image_data)
            model = client.llm.model(self._vlm_model)
            chat = lms.Chat()
            chat.add_user_message(content=prompt, images=[image_handle])
            prediction = model.respond(
                chat, response_format=response_format, config={"temperature": temperature}
            )
            if response_format is not None:
                return prediction.parsed
            return prediction.content
