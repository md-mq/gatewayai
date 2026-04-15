from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from django.http import JsonResponse  # noqa: F401
from django.http import StreamingHttpResponse  # noqa: F401

from gatewayai.providers.protocol import Provider
from gatewayai.providers.registry import create_provider

if TYPE_CHECKING:
    from gatewayai.types import CompletionRequest, Usage

logger = logging.getLogger("gatewayai.server")


def _resolve_provider(provider_name: str, request: Any) -> Provider:
    """Resolve a provider from request context.

    Credential source priority:
    1. Server-side credential pool (configured via connections or env)
    2. Request header (X-Provider-Key) — passthrough mode
    """
    api_key = _get_api_key(provider_name, request)
    return create_provider(provider_name, api_key=api_key)


def _get_api_key(provider_name: str, request: Any) -> str:
    """Extract API key from request or server config."""
    # Check request header first (passthrough mode)
    header_key = request.META.get("HTTP_X_PROVIDER_KEY")
    if header_key:
        return header_key

    # Fall back to environment variables
    import os

    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "bedrock": "AWS_ACCESS_KEY_ID",
        "vertex": "GOOGLE_APPLICATION_CREDENTIALS",
        "openai": "OPENAI_API_KEY",
        "azure": "AZURE_OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    env_var = env_map.get(provider_name, f"{provider_name.upper()}_API_KEY")
    api_key = os.environ.get(env_var, "")
    if not api_key:
        logger.warning(
            "No API key found for provider %s (checked header X-Provider-Key "
            "and env %s)",
            provider_name,
            env_var,
        )
    return api_key


async def _record_usage(
    request: Any, req: CompletionRequest, usage: Usage
) -> None:
    """Record usage asynchronously. Non-blocking.

    In Haupt, this writes to the usage tracking table.
    In EE, this feeds into billing/quota management.
    """
    logger.info(
        "Usage for request %s: input=%d output=%d",
        req.request_id,
        usage.input_tokens,
        usage.output_tokens,
    )
