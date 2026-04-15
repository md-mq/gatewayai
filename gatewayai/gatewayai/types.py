from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel


# --- Cache Control ---


class CacheControl(BaseModel):
    type: str = "ephemeral"
    ttl: str | None = None


# --- Content Blocks ---


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str
    cache_control: CacheControl | None = None


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    data: str  # Base64-encoded
    mime_type: str
    cache_control: CacheControl | None = None


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str


ContentBlock = Union[TextBlock, ImageBlock, ToolResultBlock, ThinkingBlock]


# --- Messages ---


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str  # JSON string


class Message(BaseModel):
    role: str
    content: str | list[ContentBlock]
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict


# --- Completion Request ---


class CompletionRequest(BaseModel):
    messages: list[Message]
    model: str
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    system: str | list[ContentBlock] | None = None
    thinking: bool = False
    thinking_budget: int | None = None
    cache_control: CacheControl | None = None
    request_id: str | None = None
    extra: dict[str, Any] | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())


# --- Stream Events ---


class StreamEventType(str, Enum):
    MESSAGE_START = "message_start"
    MESSAGE_END = "message_end"

    TEXT_START = "text_start"
    TEXT_DELTA = "text_delta"
    TEXT_END = "text_end"

    THINKING_START = "thinking_start"
    THINKING_DELTA = "thinking_delta"
    THINKING_END = "thinking_end"

    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"

    DONE = "done"
    ERROR = "error"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    thinking_tokens: int = 0


# --- Errors ---


class ErrorCode(str, Enum):
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    CONTEXT_LENGTH = "context_length"
    MODEL_NOT_FOUND = "model_not_found"
    CONTENT_FILTER = "content_filter"
    PROVIDER_ERROR = "provider_error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class ErrorInfo(BaseModel):
    code: ErrorCode
    message: str
    status: int | None = None
    retryable: bool = False


class StreamEvent(BaseModel):
    type: StreamEventType
    content: str | None = None
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    usage: Usage | None = None
    error: ErrorInfo | None = None
    model: str | None = None
    request_id: str | None = None
    stop_reason: str | None = None


# --- Completion Response ---


class CompletionResponse(BaseModel):
    content: str
    tool_calls: list[ToolCall] = []
    thinking: str | None = None
    usage: Usage
    model: str
    stop_reason: str | None = None
    request_id: str | None = None


# --- Model Info ---


class ModelInfo(BaseModel):
    id: str
    display_name: str
    context_window: int
    max_output_tokens: int | None = None
    supports_tools: bool = False
    supports_vision: bool = False
    supports_thinking: bool = False


# --- Exceptions ---


class ProviderError(Exception):
    def __init__(self, info: ErrorInfo):
        self.info = info
        super().__init__(info.message)


class RateLimitError(ProviderError): ...


class AuthenticationError(ProviderError): ...


class ContextLengthError(ProviderError): ...


class ModelNotFoundError(ProviderError): ...


def _error_info_to_exception(info: ErrorInfo) -> ProviderError:
    match info.code:
        case ErrorCode.RATE_LIMIT:
            return RateLimitError(info)
        case ErrorCode.AUTHENTICATION:
            return AuthenticationError(info)
        case ErrorCode.CONTEXT_LENGTH:
            return ContextLengthError(info)
        case ErrorCode.MODEL_NOT_FOUND:
            return ModelNotFoundError(info)
        case _:
            return ProviderError(info)
