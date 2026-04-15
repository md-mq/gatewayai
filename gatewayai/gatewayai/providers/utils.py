from __future__ import annotations

import asyncio

from gatewayai.types import ErrorCode, ErrorInfo, ToolDefinition


def _classify_error(e: Exception) -> ErrorInfo:
    """Classify a provider exception into a structured ErrorInfo.

    Checks for status_code attribute (httpx, anthropic, openai SDKs all set it),
    then falls back to pattern matching on the error message.
    """
    status = getattr(e, "status_code", None) or getattr(e, "status", None)
    msg = str(e).lower()

    if status == 429 or "rate" in msg:
        return ErrorInfo(
            code=ErrorCode.RATE_LIMIT,
            message=str(e),
            status=status,
            retryable=True,
        )
    if status in (401, 403):
        return ErrorInfo(
            code=ErrorCode.AUTHENTICATION, message=str(e), status=status
        )
    if status == 404:
        return ErrorInfo(
            code=ErrorCode.MODEL_NOT_FOUND, message=str(e), status=status
        )
    if ("context" in msg or "token" in msg) and status == 400:
        return ErrorInfo(
            code=ErrorCode.CONTEXT_LENGTH, message=str(e), status=status
        )
    if status in (500, 502, 503, 504):
        return ErrorInfo(
            code=ErrorCode.PROVIDER_ERROR,
            message=str(e),
            status=status,
            retryable=True,
        )
    if isinstance(e, asyncio.CancelledError):
        return ErrorInfo(code=ErrorCode.CANCELLED, message="Stream cancelled")

    return ErrorInfo(code=ErrorCode.UNKNOWN, message=str(e), status=status)


def tool_def_to_openai(tool: ToolDefinition) -> dict:
    """Convert ToolDefinition to OpenAI function calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def tool_def_to_anthropic(tool: ToolDefinition) -> dict:
    """Convert ToolDefinition to Anthropic tool format."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }
