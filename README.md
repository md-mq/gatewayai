# gatewayai

Unified LLM provider abstraction for Python. One streaming primitive, one typed event model, and pluggable adapters across Anthropic, OpenAI, Google, Azure, Bedrock, Vertex, Ollama, and DeepSeek — with an optional Django-based gateway for centralizing credentials, usage tracking, and passthrough.

## Why

Every LLM provider has its own SDK, event shape, and quirks. `gatewayai` normalizes them behind a single `Provider` protocol where `stream()` is the only primitive. Non-streaming completions are derived from streaming, so there is one code path to maintain.

- **Stream as the only primitive.** `provider.stream(request)` returns an `EventStream` you can iterate event-by-event, or `await stream.result()` for the accumulated response.
- **Typed end-to-end.** All requests, events, usage, and errors are Pydantic models.
- **Swap local for remote, invisibly.** A `GatewayProvider` implements the same `Provider` protocol and forwards requests to a remote gateway; callers don't know the difference.
- **Credential pooling & passthrough.** Optional Django server bundles credential rotation, cooldown policies, unified endpoints, and raw provider passthrough.

## Install

```bash
pip install gatewayai                # core only
pip install "gatewayai[anthropic]"   # + anthropic SDK
pip install "gatewayai[openai]"      # + openai SDK
pip install "gatewayai[google]"      # + google-genai
pip install "gatewayai[bedrock]"     # + anthropic[bedrock]
pip install "gatewayai[vertex]"      # + anthropic[vertex]
pip install "gatewayai[azure]"       # Azure OpenAI (uses openai SDK)
pip install "gatewayai[server]"      # Django server
pip install "gatewayai[all]"         # everything except server
```

Requires Python 3.9+.

## Quickstart

### Streaming

```python
import asyncio
from gatewayai import create_provider, CompletionRequest, Message, StreamEventType

async def main():
    provider = create_provider("anthropic", api_key="sk-ant-...")
    request = CompletionRequest(
        model="claude-sonnet-4-20250514",
        messages=[Message(role="user", content="Write a haiku about streaming.")],
        max_tokens=256,
    )

    async for event in provider.stream(request):
        if event.type == StreamEventType.TEXT_DELTA:
            print(event.content, end="", flush=True)

asyncio.run(main())
```

### Non-streaming (derived from streaming)

```python
from gatewayai import create_provider, complete, CompletionRequest, Message

provider = create_provider("openai", api_key="sk-...")
request = CompletionRequest(
    model="gpt-4o",
    messages=[Message(role="user", content="Summarize distributed systems in 2 sentences.")],
)
response = await complete(provider, request)
print(response.content, response.usage)
```

### Text-only streaming helper

```python
from gatewayai import stream_text

async for chunk in stream_text(provider, request):
    print(chunk, end="", flush=True)
```

### Remote gateway — same protocol, different backend

```python
from gatewayai.gateway import GatewayProvider

provider = GatewayProvider(
    base_url="https://gateway.example.com",
    token="...",
    provider="anthropic",
)
# Use the same CompletionRequest / stream() / complete() as above.
```

## Providers

Built-in adapters (lazy-imported on first use):

| Name        | Extra              | Notes                                               |
|-------------|--------------------|-----------------------------------------------------|
| `anthropic` | `[anthropic]`      | Claude via Anthropic SDK                            |
| `openai`    | `[openai]`         | OpenAI; also the base for OpenAI-compatible APIs    |
| `azure`     | `[azure]`          | Azure OpenAI                                        |
| `google`    | `[google]`         | Gemini via google-genai                             |
| `bedrock`   | `[bedrock]`        | Claude on AWS Bedrock                               |
| `vertex`    | `[vertex]`         | Claude on Google Vertex                             |
| `ollama`    | —                  | Local Ollama via OpenAI-compatible endpoint         |
| `deepseek`  | —                  | DeepSeek via OpenAI-compatible endpoint             |

Register your own:

```python
from gatewayai import register_provider

register_provider("myprovider", MyProviderFactory)
```

## Event model

Every provider emits the same `StreamEvent` shape:

- `message_start` / `message_end`
- `text_start` / `text_delta` / `text_end`
- `thinking_start` / `thinking_delta` / `thinking_end`
- `tool_call_start` / `tool_call_delta` / `tool_call_end`
- `done` (carries final `usage` and `stop_reason`)
- `error` (carries a typed `ErrorInfo` → raised as `ProviderError` subclass)

Errors are classified into `RateLimitError`, `AuthenticationError`, `ContextLengthError`, `ModelNotFoundError`, and generic `ProviderError`.

## Credential pooling

For multi-key setups, rotate across credentials with cooldowns for rate-limit / quota / outage responses:

```python
from gatewayai.credentials import (
    CredentialPool, PooledCredential, SelectionStrategy, create_pooled_provider,
)

pool = CredentialPool(
    credentials=[
        PooledCredential(provider="anthropic", api_key="key-1"),
        PooledCredential(provider="anthropic", api_key="key-2"),
    ],
    strategy=SelectionStrategy.ROUND_ROBIN,
)

provider = await create_pooled_provider(pool, "anthropic")
```

Cooldown policy is driven by upstream HTTP status (429 → 1h, 402 → 24h, 503 → 5m by default).

## Django server (optional)

With the `[server]` extra, mount the unified routes in your Django project:

```python
# urls.py
from django.urls import include, path
from gatewayai.server.urls import get_urlpatterns

urlpatterns = [
    path("streams/v1/", include(get_urlpatterns())),
]
```

Endpoints:

- `POST /streams/v1/llm/stream/` — SSE stream of `StreamEvent`s.
- `POST /streams/v1/llm/complete/` — accumulated `CompletionResponse`.
- `GET  /streams/v1/llm/models/?provider=...` — list provider models.
- `ANY  /streams/v1/llm/<provider>/<path>` — raw passthrough to the upstream, with server-side credential injection.

The server wires usage recording and credential resolution through pluggable hooks; see `gatewayai/server/`.

## Development

```bash
uv sync                  # install deps (including dev group)
uv run pytest            # run tests
uv run ruff check .      # lint
uv run mypy gatewayai    # type-check
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
