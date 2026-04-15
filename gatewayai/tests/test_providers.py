import asyncio

import pytest

from gatewayai.providers.utils import (
    _classify_error,
    tool_def_to_anthropic,
    tool_def_to_openai,
)
from gatewayai.types import ErrorCode, ToolDefinition


class TestClassifyError:
    def test_rate_limit_by_status(self):
        exc = Exception("too many requests")
        exc.status_code = 429
        info = _classify_error(exc)
        assert info.code == ErrorCode.RATE_LIMIT
        assert info.retryable

    def test_auth_error(self):
        exc = Exception("unauthorized")
        exc.status_code = 401
        info = _classify_error(exc)
        assert info.code == ErrorCode.AUTHENTICATION
        assert not info.retryable

    def test_forbidden(self):
        exc = Exception("forbidden")
        exc.status_code = 403
        info = _classify_error(exc)
        assert info.code == ErrorCode.AUTHENTICATION

    def test_not_found(self):
        exc = Exception("model not found")
        exc.status_code = 404
        info = _classify_error(exc)
        assert info.code == ErrorCode.MODEL_NOT_FOUND

    def test_server_error(self):
        exc = Exception("internal server error")
        exc.status_code = 502
        info = _classify_error(exc)
        assert info.code == ErrorCode.PROVIDER_ERROR
        assert info.retryable

    def test_cancelled(self):
        exc = asyncio.CancelledError()
        info = _classify_error(exc)
        assert info.code == ErrorCode.CANCELLED

    def test_unknown(self):
        exc = Exception("something weird")
        info = _classify_error(exc)
        assert info.code == ErrorCode.UNKNOWN

    def test_rate_limit_by_message(self):
        exc = Exception("Rate limit exceeded")
        info = _classify_error(exc)
        assert info.code == ErrorCode.RATE_LIMIT


class TestToolNormalization:
    def setup_method(self):
        self.tool = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

    def test_to_openai(self):
        result = tool_def_to_openai(self.tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"
        assert result["function"]["parameters"]["type"] == "object"

    def test_to_anthropic(self):
        result = tool_def_to_anthropic(self.tool)
        assert result["name"] == "search"
        assert result["input_schema"]["type"] == "object"
        assert "type" not in result  # No wrapping "type: function"
