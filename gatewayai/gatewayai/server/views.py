from __future__ import annotations

import json
from typing import Any

from gatewayai.server.utils import (
    JsonResponse,
    StreamingHttpResponse,
    _record_usage,
    _resolve_provider,
)
from gatewayai.types import CompletionRequest


async def stream_completion(
    request: Any, methods: dict | None = None
) -> Any:
    """SSE endpoint: streams LLM events as server-sent events.

    POST /streams/v1/llm/stream/
    Body: CompletionRequest JSON + "provider" field
    Response: text/event-stream with StreamEvent JSON per line
    """
    body = json.loads(request.body)
    provider_name = body.pop("provider")
    provider = _resolve_provider(provider_name, request)
    req = CompletionRequest(**body)

    event_stream = provider.stream(req)

    async def sse_generator():
        try:
            async for event in event_stream:
                yield f"data: {event.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        except Exception:
            event_stream.cancel()
        finally:
            usage = event_stream.usage
            if usage:
                await _record_usage(request, req, usage)

    return StreamingHttpResponse(
        sse_generator(),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-Id": req.request_id or "",
        },
    )


async def complete(request: Any, methods: dict | None = None) -> Any:
    """Non-streaming endpoint: returns the accumulated CompletionResponse.

    POST /streams/v1/llm/complete/
    Body: CompletionRequest JSON + "provider" field
    Response: CompletionResponse JSON
    """
    body = json.loads(request.body)
    provider_name = body.pop("provider")
    provider = _resolve_provider(provider_name, request)
    req = CompletionRequest(**body)

    result = await provider.stream(req).result()
    await _record_usage(request, req, result.usage)
    return JsonResponse(
        result.model_dump(),
        headers={"X-Request-Id": req.request_id or ""},
    )


async def list_models(request: Any, methods: dict | None = None) -> Any:
    """List available models for a provider.

    GET /streams/v1/llm/models/?provider=anthropic
    Response: list of ModelInfo JSON
    """
    provider_name = request.GET.get("provider")
    provider = _resolve_provider(provider_name, request)
    models = await provider.list_models()
    return JsonResponse([m.model_dump() for m in models], safe=False)
