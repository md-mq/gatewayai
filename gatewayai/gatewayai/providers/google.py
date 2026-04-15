from __future__ import annotations

import asyncio
from typing import Any

from gatewayai.event_stream import EventStream
from gatewayai.providers.utils import _classify_error
from gatewayai.types import (
    CompletionRequest,
    ErrorCode,
    ErrorInfo,
    ModelInfo,
    StreamEvent,
    StreamEventType,
)


class GoogleProvider:
    """Google Gemini provider via google-genai SDK."""

    def __init__(self, api_key: str):
        from google import genai

        self._client = genai.Client(api_key=api_key)

    @property
    def id(self) -> str:
        return "google"

    def stream(self, request: CompletionRequest) -> EventStream:
        es = EventStream()
        task = asyncio.create_task(self._run_stream(request, es))
        es.bind_task(task)
        return es

    async def _run_stream(
        self, request: CompletionRequest, es: EventStream
    ) -> None:
        raise NotImplementedError(
            "Google GenAI provider streaming not yet implemented"
        )

    async def list_models(self) -> list[ModelInfo]:
        raise NotImplementedError(
            "Google GenAI provider list_models not yet implemented"
        )

    async def close(self) -> None:
        pass
