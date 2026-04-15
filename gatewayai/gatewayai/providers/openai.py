from __future__ import annotations

import asyncio
from typing import Any

from gatewayai.event_stream import EventStream
from gatewayai.providers.utils import _classify_error
from gatewayai.types import (
    CompletionRequest,
    ContentBlock,
    ErrorCode,
    ErrorInfo,
    ImageBlock,
    ModelInfo,
    StreamEvent,
    StreamEventType,
    TextBlock,
    ToolDefinition,
    Usage,
)


class OpenAIProvider:
    """OpenAI provider via the openai SDK.

    Also serves as base for OpenAI-compatible providers
    (Ollama, DeepSeek, Groq, Together, etc.) via base_url override.
    """

    def __init__(self, api_key: str, base_url: str | None = None):
        import openai

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**kwargs)

    @property
    def id(self) -> str:
        return "openai"

    def stream(self, request: CompletionRequest) -> EventStream:
        es = EventStream()
        task = asyncio.create_task(self._run_stream(request, es))
        es.bind_task(task)
        return es

    async def _run_stream(
        self, request: CompletionRequest, es: EventStream
    ) -> None:
        try:
            kwargs = self._build_params(request)
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}

            es.push(
                StreamEvent(
                    type=StreamEventType.MESSAGE_START,
                    model=request.model,
                    request_id=request.request_id,
                )
            )

            last_usage: Usage | None = None
            stop_reason: str | None = None

            response = await self._client.chat.completions.create(**kwargs)
            async for chunk in response:
                for se in self._convert_chunk(chunk):
                    es.push(se)
                if chunk.usage:
                    last_usage = Usage(
                        input_tokens=chunk.usage.prompt_tokens or 0,
                        output_tokens=chunk.usage.completion_tokens or 0,
                        cache_read_tokens=getattr(
                            getattr(
                                chunk.usage, "prompt_tokens_details", None
                            ),
                            "cached_tokens",
                            0,
                        )
                        or 0,
                    )
                if chunk.choices and chunk.choices[0].finish_reason:
                    stop_reason = chunk.choices[0].finish_reason

            es.push(
                StreamEvent(
                    type=StreamEventType.DONE,
                    usage=last_usage or Usage(),
                    stop_reason=stop_reason,
                )
            )
        except asyncio.CancelledError:
            es.push(
                StreamEvent(
                    type=StreamEventType.ERROR,
                    error=ErrorInfo(
                        code=ErrorCode.CANCELLED, message="Stream cancelled"
                    ),
                )
            )
        except Exception as e:
            es.push(
                StreamEvent(
                    type=StreamEventType.ERROR, error=_classify_error(e)
                )
            )

    def _build_params(self, request: CompletionRequest) -> dict:
        messages = self._convert_messages(request.messages)
        if request.system:
            if isinstance(request.system, str):
                messages.insert(
                    0, {"role": "system", "content": request.system}
                )
            else:
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": [
                            self._content_block_to_dict(b)
                            for b in request.system
                        ],
                    },
                )

        params: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
        }
        if request.tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]
        if request.tool_choice:
            params["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if request.stop:
            params["stop"] = request.stop

        if request.extra:
            params.update(request.extra)

        return params

    def _convert_messages(self, messages: list) -> list[dict]:
        result = []
        for msg in messages:
            d: dict[str, Any] = {"role": msg.role}
            if isinstance(msg.content, str):
                d["content"] = msg.content
            else:
                d["content"] = [
                    self._content_block_to_dict(b) for b in msg.content
                ]
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
            result.append(d)
        return result

    def _content_block_to_dict(self, block: ContentBlock) -> dict:
        if isinstance(block, TextBlock):
            return {"type": "text", "text": block.text}
        if isinstance(block, ImageBlock):
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{block.mime_type};base64,{block.data}"
                },
            }
        return block.model_dump()

    def _convert_chunk(self, chunk: Any) -> list[StreamEvent]:
        """Convert OpenAI ChatCompletionChunk to normalized StreamEvents."""
        results: list[StreamEvent] = []
        if not chunk.choices:
            return results

        choice = chunk.choices[0]
        delta = choice.delta

        if delta.content:
            results.append(
                StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    content=delta.content,
                )
            )

        if delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function and tc.function.name:
                    results.append(
                        StreamEvent(
                            type=StreamEventType.TOOL_CALL_START,
                            tool_call_id=tc.id,
                            tool_call_name=tc.function.name,
                        )
                    )
                if tc.function and tc.function.arguments:
                    results.append(
                        StreamEvent(
                            type=StreamEventType.TOOL_CALL_DELTA,
                            tool_call_id=tc.id,
                            content=tc.function.arguments,
                        )
                    )

        return results

    async def list_models(self) -> list[ModelInfo]:
        result = await self._client.models.list()
        return [
            ModelInfo(
                id=m.id, display_name=m.id, context_window=128000
            )
            for m in result.data
        ]

    async def close(self) -> None:
        await self._client.close()
