import asyncio

import pytest

from gatewayai.event_stream import EventStream, _ResponseAccumulator
from gatewayai.types import (
    ErrorCode,
    ErrorInfo,
    ProviderError,
    StreamEvent,
    StreamEventType,
    ToolCall,
    Usage,
)


class TestResponseAccumulator:
    def test_text_accumulation(self):
        acc = _ResponseAccumulator()
        acc.add(StreamEvent(type=StreamEventType.MESSAGE_START, model="m1", request_id="r1"))
        acc.add(StreamEvent(type=StreamEventType.TEXT_DELTA, content="hello "))
        acc.add(StreamEvent(type=StreamEventType.TEXT_DELTA, content="world"))
        acc.add(StreamEvent(type=StreamEventType.DONE, usage=Usage(input_tokens=10, output_tokens=2), stop_reason="end_turn"))

        result = acc.build()
        assert result.content == "hello world"
        assert result.model == "m1"
        assert result.request_id == "r1"
        assert result.usage.input_tokens == 10
        assert result.stop_reason == "end_turn"

    def test_thinking_accumulation(self):
        acc = _ResponseAccumulator()
        acc.add(StreamEvent(type=StreamEventType.MESSAGE_START, model="m1"))
        acc.add(StreamEvent(type=StreamEventType.THINKING_DELTA, content="step 1, "))
        acc.add(StreamEvent(type=StreamEventType.THINKING_DELTA, content="step 2"))
        acc.add(StreamEvent(type=StreamEventType.TEXT_DELTA, content="answer"))
        acc.add(StreamEvent(type=StreamEventType.DONE, usage=Usage()))

        result = acc.build()
        assert result.thinking == "step 1, step 2"
        assert result.content == "answer"

    def test_tool_call_accumulation(self):
        acc = _ResponseAccumulator()
        acc.add(StreamEvent(type=StreamEventType.MESSAGE_START, model="m1"))
        acc.add(StreamEvent(type=StreamEventType.TOOL_CALL_START, tool_call_id="tc1", tool_call_name="search"))
        acc.add(StreamEvent(type=StreamEventType.TOOL_CALL_DELTA, tool_call_id="tc1", content='{"q":'))
        acc.add(StreamEvent(type=StreamEventType.TOOL_CALL_DELTA, tool_call_id="tc1", content='"test"}'))
        acc.add(StreamEvent(type=StreamEventType.TOOL_CALL_END, tool_call_id="tc1"))
        acc.add(StreamEvent(type=StreamEventType.DONE, usage=Usage(), stop_reason="tool_use"))

        result = acc.build()
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == '{"q":"test"}'
        assert result.stop_reason == "tool_use"

    def test_empty_thinking_is_none(self):
        acc = _ResponseAccumulator()
        acc.add(StreamEvent(type=StreamEventType.MESSAGE_START, model="m1"))
        acc.add(StreamEvent(type=StreamEventType.TEXT_DELTA, content="hi"))
        acc.add(StreamEvent(type=StreamEventType.DONE, usage=Usage()))

        result = acc.build()
        assert result.thinking is None


class TestEventStream:
    @pytest.mark.asyncio
    async def test_iterate_events(self):
        es = EventStream()

        # Simulate a producer
        async def producer():
            es.push(StreamEvent(type=StreamEventType.MESSAGE_START, model="m1"))
            es.push(StreamEvent(type=StreamEventType.TEXT_DELTA, content="hi"))
            es.push(StreamEvent(type=StreamEventType.DONE, usage=Usage()))

        task = asyncio.create_task(producer())
        es.bind_task(task)

        events = []
        async for event in es:
            events.append(event)

        assert len(events) == 3
        assert events[0].type == StreamEventType.MESSAGE_START
        assert events[1].type == StreamEventType.TEXT_DELTA
        assert events[2].type == StreamEventType.DONE

    @pytest.mark.asyncio
    async def test_result(self):
        es = EventStream()

        async def producer():
            es.push(StreamEvent(type=StreamEventType.MESSAGE_START, model="m1", request_id="r1"))
            es.push(StreamEvent(type=StreamEventType.TEXT_DELTA, content="hello"))
            es.push(StreamEvent(type=StreamEventType.DONE, usage=Usage(input_tokens=5, output_tokens=1)))

        task = asyncio.create_task(producer())
        es.bind_task(task)

        result = await es.result()
        assert result.content == "hello"
        assert result.model == "m1"
        assert result.usage.input_tokens == 5

    @pytest.mark.asyncio
    async def test_error_raises_on_result(self):
        es = EventStream()

        async def producer():
            es.push(StreamEvent(type=StreamEventType.MESSAGE_START, model="m1"))
            es.push(
                StreamEvent(
                    type=StreamEventType.ERROR,
                    error=ErrorInfo(code=ErrorCode.RATE_LIMIT, message="rate limited", status=429),
                )
            )

        task = asyncio.create_task(producer())
        es.bind_task(task)

        with pytest.raises(ProviderError) as exc_info:
            await es.result()
        assert exc_info.value.info.code == ErrorCode.RATE_LIMIT

    @pytest.mark.asyncio
    async def test_cancel(self):
        es = EventStream()

        async def slow_producer():
            es.push(StreamEvent(type=StreamEventType.MESSAGE_START, model="m1"))
            await asyncio.sleep(10)  # Will be cancelled

        task = asyncio.create_task(slow_producer())
        es.bind_task(task)

        # Wait for MESSAGE_START
        event = await es.__anext__()
        assert event.type == StreamEventType.MESSAGE_START

        # Cancel
        es.cancel()

        with pytest.raises(ProviderError) as exc_info:
            await es.result()
        assert exc_info.value.info.code == ErrorCode.CANCELLED

    @pytest.mark.asyncio
    async def test_usage_property(self):
        es = EventStream()

        async def producer():
            es.push(StreamEvent(type=StreamEventType.MESSAGE_START, model="m1"))
            es.push(StreamEvent(type=StreamEventType.TEXT_DELTA, content="hi"))
            es.push(StreamEvent(type=StreamEventType.DONE, usage=Usage(input_tokens=10, output_tokens=2)))

        task = asyncio.create_task(producer())
        es.bind_task(task)

        await es.result()

        usage = es.usage
        assert usage is not None
        assert usage.input_tokens == 10
        assert usage.output_tokens == 2

    @pytest.mark.asyncio
    async def test_usage_none_before_done(self):
        es = EventStream()
        assert es.usage is None
