from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gatewayai.event_stream import EventStream
    from gatewayai.types import (
        CompletionRequest,
        CompletionResponse,
        ModelInfo,
        StreamEventType,
    )


@runtime_checkable
class Provider(Protocol):
    """Every LLM provider implements this protocol.

    One method: stream(). That's it.
    complete() is derived — see convenience function below.
    """

    @property
    def id(self) -> str: ...

    def stream(self, request: CompletionRequest) -> EventStream: ...

    async def list_models(self) -> list[ModelInfo]: ...


async def complete(
    provider: Provider, request: CompletionRequest
) -> CompletionResponse:
    """Non-streaming completion. Syntactic sugar for stream().result()."""
    return await provider.stream(request).result()


async def stream_text(
    provider: Provider, request: CompletionRequest
) -> AsyncIterator[str]:
    """Yield only text deltas."""
    from gatewayai.types import StreamEventType

    async for event in provider.stream(request):
        if event.type == StreamEventType.TEXT_DELTA and event.content:
            yield event.content
