from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from gatewayai.types import (
    CompletionResponse,
    ErrorCode,
    ErrorInfo,
    StreamEvent,
    StreamEventType,
    ToolCall,
    Usage,
    _error_info_to_exception,
)


class _ResponseAccumulator:
    """Accumulates StreamEvents into a CompletionResponse."""

    def __init__(self) -> None:
        self._text_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._tool_calls: dict[str, ToolCall] = {}
        self._current_tool_args: dict[str, list[str]] = {}
        self._usage: Usage = Usage()
        self._model: str = ""
        self._stop_reason: str | None = None
        self._request_id: str | None = None

    def add(self, event: StreamEvent) -> None:
        match event.type:
            case StreamEventType.MESSAGE_START:
                self._model = event.model or ""
                self._request_id = event.request_id
            case StreamEventType.TEXT_DELTA:
                self._text_parts.append(event.content or "")
            case StreamEventType.THINKING_DELTA:
                self._thinking_parts.append(event.content or "")
            case StreamEventType.TOOL_CALL_START:
                if event.tool_call_id:
                    self._tool_calls[event.tool_call_id] = ToolCall(
                        id=event.tool_call_id,
                        name=event.tool_call_name or "",
                        arguments="",
                    )
                    self._current_tool_args[event.tool_call_id] = []
            case StreamEventType.TOOL_CALL_DELTA:
                if event.tool_call_id and event.content:
                    self._current_tool_args.setdefault(
                        event.tool_call_id, []
                    ).append(event.content)
            case StreamEventType.TOOL_CALL_END:
                if (
                    event.tool_call_id
                    and event.tool_call_id in self._tool_calls
                ):
                    self._tool_calls[event.tool_call_id].arguments = "".join(
                        self._current_tool_args.get(event.tool_call_id, [])
                    )
            case StreamEventType.DONE:
                if event.usage:
                    self._usage = event.usage
                if event.stop_reason:
                    self._stop_reason = event.stop_reason

    def build(self) -> CompletionResponse:
        return CompletionResponse(
            content="".join(self._text_parts),
            tool_calls=list(self._tool_calls.values()),
            thinking="".join(self._thinking_parts) or None,
            usage=self._usage,
            model=self._model,
            stop_reason=self._stop_reason,
            request_id=self._request_id,
        )


class EventStream:
    """Async iterable of StreamEvents with a result promise.

    Consumers choose their interface:
    - Streaming: ``async for event in stream: ...``
    - Non-streaming: ``result = await stream.result()``
    - Mixed: iterate some events, then await result for the rest
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._result_future: asyncio.Future[CompletionResponse] = (
            asyncio.get_running_loop().create_future()
        )
        self._accumulator = _ResponseAccumulator()
        self._task: asyncio.Task | None = None

    def bind_task(self, task: asyncio.Task) -> None:
        """Bind the producer task so cancel() can stop it."""
        self._task = task

    def push(self, event: StreamEvent) -> None:
        """Called by the provider to emit an event."""
        self._accumulator.add(event)
        self._queue.put_nowait(event)

        if event.type == StreamEventType.DONE:
            if not self._result_future.done():
                self._result_future.set_result(self._accumulator.build())
            self._queue.put_nowait(None)  # Sentinel
        elif event.type == StreamEventType.ERROR:
            exc = _error_info_to_exception(
                event.error
                or ErrorInfo(code=ErrorCode.UNKNOWN, message="Unknown error")
            )
            if not self._result_future.done():
                self._result_future.set_exception(exc)
            self._queue.put_nowait(None)

    def cancel(self) -> None:
        """Cancel the in-progress stream.

        Cancels the producer task and pushes a CANCELLED error event.
        Safe to call multiple times or after stream completion.
        """
        if self._task and not self._task.done():
            self._task.cancel()
        if not self._result_future.done():
            info = ErrorInfo(
                code=ErrorCode.CANCELLED, message="Stream cancelled by client"
            )
            self.push(
                StreamEvent(type=StreamEventType.ERROR, error=info)
            )

    @property
    def usage(self) -> Usage | None:
        """Accumulated usage so far. Available after DONE event."""
        built = self._accumulator._usage
        if built.input_tokens == 0 and built.output_tokens == 0:
            return None
        return built

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self

    async def __anext__(self) -> StreamEvent:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def result(self) -> CompletionResponse:
        """Await the final accumulated response.

        Raises ProviderError (or subclass) if the stream ends with an error.
        """
        return await self._result_future
