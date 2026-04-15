import pytest

from gatewayai.types import (
    CacheControl,
    CompletionRequest,
    CompletionResponse,
    ErrorCode,
    ErrorInfo,
    ImageBlock,
    Message,
    ModelInfo,
    ProviderError,
    RateLimitError,
    StreamEvent,
    StreamEventType,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
    Usage,
    _error_info_to_exception,
)


class TestContentBlocks:
    def test_text_block(self):
        block = TextBlock(text="hello")
        assert block.type == "text"
        assert block.text == "hello"
        assert block.cache_control is None

    def test_text_block_with_cache(self):
        block = TextBlock(
            text="cached", cache_control=CacheControl(type="ephemeral")
        )
        assert block.cache_control.type == "ephemeral"

    def test_image_block(self):
        block = ImageBlock(data="base64data", mime_type="image/png")
        assert block.type == "image"

    def test_tool_result_block(self):
        block = ToolResultBlock(
            tool_use_id="123", content="result", is_error=False
        )
        assert block.type == "tool_result"
        assert not block.is_error

    def test_thinking_block(self):
        block = ThinkingBlock(thinking="reasoning...")
        assert block.type == "thinking"


class TestMessage:
    def test_simple_message(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.tool_calls is None

    def test_message_with_blocks(self):
        msg = Message(
            role="user",
            content=[TextBlock(text="hello"), ImageBlock(data="x", mime_type="image/png")],
        )
        assert len(msg.content) == 2

    def test_message_with_tool_calls(self):
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="1", name="search", arguments='{"q":"test"}')],
        )
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"


class TestCompletionRequest:
    def test_auto_request_id(self):
        req = CompletionRequest(
            messages=[Message(role="user", content="hi")],
            model="claude-sonnet-4-20250514",
        )
        assert req.request_id is not None
        assert len(req.request_id) == 36  # UUID format

    def test_explicit_request_id(self):
        req = CompletionRequest(
            messages=[Message(role="user", content="hi")],
            model="gpt-4o",
            request_id="custom-id",
        )
        assert req.request_id == "custom-id"

    def test_with_tools(self):
        req = CompletionRequest(
            messages=[Message(role="user", content="hi")],
            model="claude-sonnet-4-20250514",
            tools=[
                ToolDefinition(
                    name="search",
                    description="Search the web",
                    parameters={"type": "object", "properties": {}},
                )
            ],
        )
        assert len(req.tools) == 1

    def test_with_system(self):
        req = CompletionRequest(
            messages=[Message(role="user", content="hi")],
            model="claude-sonnet-4-20250514",
            system="You are a helpful assistant",
        )
        assert req.system == "You are a helpful assistant"

    def test_with_extra(self):
        req = CompletionRequest(
            messages=[Message(role="user", content="hi")],
            model="gpt-4o",
            extra={"seed": 42},
        )
        assert req.extra["seed"] == 42


class TestStreamEvent:
    def test_text_delta(self):
        event = StreamEvent(
            type=StreamEventType.TEXT_DELTA, content="hello"
        )
        assert event.type == StreamEventType.TEXT_DELTA
        assert event.content == "hello"

    def test_done_with_usage(self):
        event = StreamEvent(
            type=StreamEventType.DONE,
            usage=Usage(input_tokens=100, output_tokens=50),
            stop_reason="end_turn",
        )
        assert event.usage.input_tokens == 100
        assert event.stop_reason == "end_turn"

    def test_error_event(self):
        event = StreamEvent(
            type=StreamEventType.ERROR,
            error=ErrorInfo(
                code=ErrorCode.RATE_LIMIT,
                message="Too many requests",
                status=429,
                retryable=True,
            ),
        )
        assert event.error.retryable

    def test_serialization_roundtrip(self):
        event = StreamEvent(
            type=StreamEventType.TEXT_DELTA, content="test"
        )
        json_str = event.model_dump_json()
        restored = StreamEvent.model_validate_json(json_str)
        assert restored.type == StreamEventType.TEXT_DELTA
        assert restored.content == "test"


class TestErrors:
    def test_error_info_to_rate_limit(self):
        info = ErrorInfo(
            code=ErrorCode.RATE_LIMIT, message="rate limited", status=429
        )
        exc = _error_info_to_exception(info)
        assert isinstance(exc, RateLimitError)
        assert exc.info.status == 429

    def test_error_info_to_provider_error(self):
        info = ErrorInfo(
            code=ErrorCode.PROVIDER_ERROR, message="server error", status=500
        )
        exc = _error_info_to_exception(info)
        assert isinstance(exc, ProviderError)
        assert not isinstance(exc, RateLimitError)

    def test_error_info_to_unknown(self):
        info = ErrorInfo(code=ErrorCode.UNKNOWN, message="???")
        exc = _error_info_to_exception(info)
        assert isinstance(exc, ProviderError)


class TestModelInfo:
    def test_basic(self):
        info = ModelInfo(
            id="claude-sonnet-4-20250514",
            display_name="Claude Sonnet 4",
            context_window=200_000,
            supports_tools=True,
        )
        assert info.context_window == 200_000
        assert info.supports_tools
        assert not info.supports_thinking
