from __future__ import annotations

import logging
from typing import Any

import httpx

from django.http import HttpResponse, StreamingHttpResponse

from gatewayai.credentials import (
    COOLDOWN_POLICY,
    CredentialPool,
    PooledCredential,
)
from gatewayai.server.utils import _get_api_key
from gatewayai.types import ErrorCode, ErrorInfo

logger = logging.getLogger("gatewayai.server")

UPSTREAM_FORMATS: dict[str, dict[str, str]] = {
    "openai": {
        "auth_header": "Authorization",
        "auth_format": "Bearer {api_key}",
    },
    "anthropic": {
        "auth_header": "x-api-key",
        "auth_format": "{api_key}",
    },
    "google": {
        "auth_header": "x-goog-api-key",
        "auth_format": "{api_key}",
    },
}

_UPSTREAMS: dict[str, dict[str, str]] = {
    "openai": {"base_url": "https://api.openai.com", "format": "openai"},
    "anthropic": {"base_url": "https://api.anthropic.com", "format": "anthropic"},
    "google": {"base_url": "https://generativelanguage.googleapis.com", "format": "google"},
}


def register_upstream(
    name: str,
    base_url: str,
    format: str = "openai",
) -> None:
    """Register an upstream provider for passthrough.

    Most providers are OpenAI-compatible and just need a base_url:
        register_upstream("deepseek", "https://api.deepseek.com")
        register_upstream("groq", "https://api.groq.com/openai")
        register_upstream("together", "https://api.together.xyz")
    """
    if format not in UPSTREAM_FORMATS:
        raise ValueError(
            f"Unknown format {format!r}. Must be one of: {list(UPSTREAM_FORMATS)}"
        )
    _UPSTREAMS[name] = {"base_url": base_url.rstrip("/"), "format": format}


def get_upstream(name: str) -> dict[str, str] | None:
    """Get upstream config for a provider."""
    return _UPSTREAMS.get(name)


def list_upstreams() -> list[str]:
    """List registered upstream provider names."""
    return list(_UPSTREAMS)


AUTH_HEADERS_TO_STRIP = frozenset({
    "authorization",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
})

PROPAGATED_RESPONSE_HEADERS = (
    "x-request-id",
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "anthropic-ratelimit-requests-limit",
    "anthropic-ratelimit-requests-remaining",
    "anthropic-ratelimit-requests-reset",
    "anthropic-ratelimit-tokens-limit",
    "anthropic-ratelimit-tokens-remaining",
    "anthropic-ratelimit-tokens-reset",
)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
            follow_redirects=True,
        )
    return _client


def _build_upstream_headers(
    request: Any,
    upstream_config: dict[str, str],
    api_key: str,
) -> dict[str, str]:
    """Build headers for the upstream request.

    Extracts headers from Django request.META, strips auth and host headers,
    injects the provider-specific auth header.
    """
    fmt = UPSTREAM_FORMATS[upstream_config["format"]]
    headers: dict[str, str] = {}

    for key, value in request.META.items():
        if key.startswith("HTTP_"):
            header_name = key[5:].replace("_", "-").lower()
            if header_name not in AUTH_HEADERS_TO_STRIP and header_name != "host":
                headers[header_name] = value

    content_type = request.META.get("CONTENT_TYPE")
    if content_type:
        headers["content-type"] = content_type

    headers[fmt["auth_header"].lower()] = fmt["auth_format"].format(
        api_key=api_key
    )

    return headers


def _resolve_credential(
    provider: str,
    request: Any,
    pool: CredentialPool | None = None,
) -> PooledCredential | None:
    """Resolve a credential for passthrough.

    Priority:
    1. Server-side CredentialPool (if provided — multi-key rotation)
    2. Single key from _get_api_key() (header or env fallback)
    """
    if pool is not None:
        return pool.select(provider)

    api_key = _get_api_key(provider, request)
    if not api_key:
        return None

    return PooledCredential(provider=provider, api_key=api_key)


_pool: CredentialPool | None = None


def configure_pool(pool: CredentialPool) -> None:
    global _pool
    _pool = pool


async def passthrough(
    request: Any,
    provider: str,
    path: str,
    methods: dict | None = None,
) -> Any:
    """Raw passthrough proxy — forwards requests to upstream LLM providers.

    Catches all paths under llm/<provider>/<path> and forwards them
    to the provider's API with auth header swap and credential pooling.
    """
    upstream = get_upstream(provider)
    if upstream is None:
        return HttpResponse(
            f'{{"error": "Unknown provider: {provider}", '
            f'"available": {list_upstreams()}}}',
            status=404,
            content_type="application/json",
        )

    cred = _resolve_credential(provider, request, _pool)
    if cred is None:
        if _pool is not None and _pool.all_exhausted(provider):
            return HttpResponse(
                '{"error": "All API keys exhausted for provider"}',
                status=429,
                content_type="application/json",
            )
        return HttpResponse(
            f'{{"error": "No API key configured for provider: {provider}"}}',
            status=401,
            content_type="application/json",
        )

    upstream_url = f"{upstream['base_url']}/{path}"
    query_string = request.META.get("QUERY_STRING")
    if query_string:
        upstream_url += f"?{query_string}"

    headers = _build_upstream_headers(request, upstream, cred.api_key)
    method = request.method
    body = request.body if method in ("POST", "PUT", "PATCH") else None

    client = _get_client()

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            upstream_req = client.build_request(
                method, upstream_url, headers=headers, content=body
            )
            upstream_resp = await client.send(upstream_req, stream=True)

            if upstream_resp.status_code in COOLDOWN_POLICY and _pool is not None:
                await upstream_resp.aclose()
                error_info = ErrorInfo(
                    code=ErrorCode.RATE_LIMIT
                    if upstream_resp.status_code == 429
                    else ErrorCode.PROVIDER_ERROR,
                    message=f"Upstream returned {upstream_resp.status_code}",
                    status=upstream_resp.status_code,
                )
                _pool.report_error(cred, error_info)
                logger.warning(
                    "Passthrough %s: key cooled down (status %d), attempt %d/%d",
                    provider,
                    upstream_resp.status_code,
                    attempt + 1,
                    max_attempts,
                )

                cred = _pool.select(provider)
                if cred is None:
                    return HttpResponse(
                        '{"error": "All API keys exhausted for provider"}',
                        status=429,
                        content_type="application/json",
                    )
                headers = _build_upstream_headers(request, upstream, cred.api_key)
                continue

            return await _build_response(upstream_resp)

        except httpx.ConnectError:
            logger.error("Passthrough %s: upstream connection failed", provider)
            return HttpResponse(
                '{"error": "Upstream connection failed"}',
                status=502,
                content_type="application/json",
            )
        except httpx.TimeoutException:
            logger.error("Passthrough %s: upstream timeout", provider)
            return HttpResponse(
                '{"error": "Upstream timeout"}',
                status=504,
                content_type="application/json",
            )

    return HttpResponse(
        '{"error": "All retry attempts exhausted"}',
        status=429,
        content_type="application/json",
    )


async def _build_response(upstream_resp: httpx.Response) -> HttpResponse:
    """Build Django response from upstream httpx response."""
    content_type = upstream_resp.headers.get("content-type", "application/json")
    is_streaming = (
        "text/event-stream" in content_type
        or "application/x-ndjson" in content_type
    )

    if is_streaming:

        async def relay():
            try:
                async for chunk in upstream_resp.aiter_bytes(4096):
                    yield chunk
            finally:
                await upstream_resp.aclose()

        response = StreamingHttpResponse(
            relay(),
            status=upstream_resp.status_code,
            content_type=content_type,
        )
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
    else:
        body = await upstream_resp.aread()
        await upstream_resp.aclose()
        response = HttpResponse(
            body,
            status=upstream_resp.status_code,
            content_type=content_type,
        )

    for header in PROPAGATED_RESPONSE_HEADERS:
        value = upstream_resp.headers.get(header)
        if value:
            response[header] = value

    return response
