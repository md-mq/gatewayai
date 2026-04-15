from __future__ import annotations

import asyncio

import httpx

from gatewayai.event_stream import EventStream
from gatewayai.types import (
    CompletionRequest,
    ErrorCode,
    ErrorInfo,
    ModelInfo,
    StreamEvent,
    StreamEventType,
)


class GatewayProvider:
    """Provider that calls a remote gateway (Haupt) instead of local SDKs.

    Same Protocol as local providers. Consumers don't know the difference.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        provider: str,
        timeout: float = 300.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._provider = provider
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"token {token}"},
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    @property
    def id(self) -> str:
        return f"gateway:{self._provider}"

    def stream(self, request: CompletionRequest) -> EventStream:
        es = EventStream()
        task = asyncio.create_task(self._run_stream(request, es))
        es.bind_task(task)
        return es

    async def _run_stream(
        self, request: CompletionRequest, es: EventStream
    ) -> None:
        try:
            payload = request.model_dump()
            payload["provider"] = self._provider

            async with self._client.stream(
                "POST",
                "/streams/v1/llm/stream/",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    event = StreamEvent.model_validate_json(data)
                    es.push(event)

            # If stream ended without DONE event, push one
            if not es._result_future.done():
                es.push(StreamEvent(type=StreamEventType.DONE))

        except asyncio.CancelledError:
            pass
        except httpx.HTTPStatusError as e:
            es.push(
                StreamEvent(
                    type=StreamEventType.ERROR,
                    error=ErrorInfo(
                        code=ErrorCode.PROVIDER_ERROR,
                        message=f"Gateway error: {e.response.text}",
                        status=e.response.status_code,
                        retryable=e.response.status_code
                        in (502, 503, 504),
                    ),
                )
            )
        except Exception as e:
            es.push(
                StreamEvent(
                    type=StreamEventType.ERROR,
                    error=ErrorInfo(
                        code=ErrorCode.UNKNOWN, message=str(e)
                    ),
                )
            )

    async def list_models(self) -> list[ModelInfo]:
        resp = await self._client.get(
            "/streams/v1/llm/models/",
            params={"provider": self._provider},
        )
        resp.raise_for_status()
        return [ModelInfo(**m) for m in resp.json()]

    async def close(self) -> None:
        await self._client.aclose()
