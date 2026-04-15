from __future__ import annotations

import asyncio
from typing import Any

from gatewayai.event_stream import EventStream
from gatewayai.providers.utils import _classify_error
from gatewayai.types import (
    CacheControl,
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


class AnthropicProvider:
    """Anthropic Claude provider via the anthropic SDK."""

    def __init__(self, api_key: str, base_url: str | None = None):
        import anthropic

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)

    @property
    def id(self) -> str:
        return "anthropic"

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
            async with self._client.messages.stream(**kwargs) as stream:
                es.push(
                    StreamEvent(
                        type=StreamEventType.MESSAGE_START,
                        model=request.model,
                        request_id=request.request_id,
                    )
                )

                async for event in stream:
                    for se in self._convert_event(event):
                        es.push(se)

                final = await stream.get_final_message()
                es.push(
                    StreamEvent(
                        type=StreamEventType.DONE,
                        stop_reason=final.stop_reason,
                        usage=Usage(
                            input_tokens=final.usage.input_tokens,
                            output_tokens=final.usage.output_tokens,
                            cache_read_tokens=getattr(
                                final.usage,
                                "cache_read_input_tokens",
                                0,
                            )
                            or 0,
                            cache_creation_tokens=getattr(
                                final.usage,
                                "cache_creation_input_tokens",
                                0,
                            )
                            or 0,
                        ),
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
        params: dict[str, Any] = {
            "model": request.model,
            "messages": self._convert_messages(request.messages),
            "max_tokens": request.max_tokens or 4096,
        }

        if request.system:
            params["system"] = self._convert_system(
                request.system, request.cache_control
            )

        if request.tools:
            params["tools"] = self._convert_tools(request.tools)
        if request.tool_choice:
            params["tool_choice"] = self._convert_tool_choice(
                request.tool_choice
            )
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.thinking:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": request.thinking_budget or 10000,
            }

        if request.extra:
            extra = dict(request.extra)
            extra_headers = extra.pop("extra_headers", None)
            params.update(extra)
            if extra_headers:
                params["extra_headers"] = extra_headers

        return params

    def _convert_system(
        self,
        system: str | list[ContentBlock],
        cache_control: CacheControl | None,
    ) -> str | list[dict]:
        if isinstance(system, str):
            if cache_control:
                return [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": cache_control.type},
                    }
                ]
            return system

        blocks = [self._content_block_to_dict(b) for b in system]
        if cache_control and blocks:
            last = blocks[-1]
            if "cache_control" not in last:
                last["cache_control"] = {"type": cache_control.type}
        return blocks

    def _content_block_to_dict(self, block: ContentBlock) -> dict:
        if isinstance(block, TextBlock):
            d: dict[str, Any] = {"type": "text", "text": block.text}
            if block.cache_control:
                d["cache_control"] = {"type": block.cache_control.type}
            return d
        if isinstance(block, ImageBlock):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": block.mime_type,
                    "data": block.data,
                },
            }
        return block.model_dump()

    def _convert_messages(self, messages: list) -> list[dict]:
        from gatewayai.types import Message

        result = []
        for msg in messages:
            if msg.role == "system":
                continue
            d: dict[str, Any] = {"role": msg.role}
            if isinstance(msg.content, str):
                d["content"] = msg.content
            else:
                d["content"] = [
                    self._content_block_to_dict(b) for b in msg.content
                ]
            if msg.tool_calls:
                content = d.get("content", [])
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                d["content"] = content
            result.append(d)
        return result

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    def _convert_tool_choice(self, tool_choice: str | dict) -> dict | str:
        if isinstance(tool_choice, dict):
            return tool_choice
        if tool_choice == "required":
            return {"type": "any"}
        if tool_choice == "none":
            return {"type": "none"}
        return {"type": "auto"}

    def _convert_event(self, event: Any) -> list[StreamEvent]:
        """Convert an Anthropic stream event to normalized StreamEvents."""
        results: list[StreamEvent] = []
        event_type = getattr(event, "type", None)

        if event_type == "content_block_start":
            block = event.content_block
            block_type = getattr(block, "type", None)
            if block_type == "text":
                results.append(
                    StreamEvent(type=StreamEventType.TEXT_START)
                )
            elif block_type == "tool_use":
                results.append(
                    StreamEvent(
                        type=StreamEventType.TOOL_CALL_START,
                        tool_call_id=block.id,
                        tool_call_name=block.name,
                    )
                )
            elif block_type == "thinking":
                results.append(
                    StreamEvent(type=StreamEventType.THINKING_START)
                )

        elif event_type == "content_block_delta":
            delta = event.delta
            delta_type = getattr(delta, "type", None)
            if delta_type == "text_delta":
                results.append(
                    StreamEvent(
                        type=StreamEventType.TEXT_DELTA,
                        content=delta.text,
                    )
                )
            elif delta_type == "input_json_delta":
                results.append(
                    StreamEvent(
                        type=StreamEventType.TOOL_CALL_DELTA,
                        tool_call_id=getattr(event, "tool_call_id", None),
                        content=delta.partial_json,
                    )
                )
            elif delta_type == "thinking_delta":
                results.append(
                    StreamEvent(
                        type=StreamEventType.THINKING_DELTA,
                        content=delta.thinking,
                    )
                )

        elif event_type == "content_block_stop":
            # Determine which end event to emit based on accumulated state.
            # The Anthropic SDK doesn't tell us the block type on stop,
            # so we emit a generic TEXT_END. The accumulator handles it.
            results.append(StreamEvent(type=StreamEventType.TEXT_END))

        return results

    async def list_models(self) -> list[ModelInfo]:
        result = await self._client.models.list()
        return [
            ModelInfo(
                id=m.id,
                display_name=getattr(m, "display_name", m.id),
                context_window=getattr(m, "context_window", 200000),
            )
            for m in result.data
        ]

    async def close(self) -> None:
        await self._client.close()
