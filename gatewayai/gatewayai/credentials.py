from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from gatewayai.providers.protocol import Provider
from gatewayai.providers.registry import create_provider
from gatewayai.types import (
    CompletionRequest,
    ErrorInfo,
    Message,
    ModelInfo,
    Usage,
)


class SelectionStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_USED = "least_used"
    FILL_FIRST = "fill_first"


@dataclass
class PooledCredential:
    provider: str
    api_key: str
    base_url: str | None = None
    priority: int = 0
    use_count: int = 0
    cooldown_until: datetime | None = None
    extra: dict | None = None

    @property
    def available(self) -> bool:
        return (
            self.cooldown_until is None
            or datetime.now() >= self.cooldown_until
        )


COOLDOWN_POLICY: dict[int, timedelta] = {
    429: timedelta(hours=1),
    402: timedelta(hours=24),
    503: timedelta(minutes=5),
}


class CredentialPool:
    def __init__(
        self,
        credentials: list[PooledCredential],
        strategy: SelectionStrategy = SelectionStrategy.ROUND_ROBIN,
    ):
        self._credentials = credentials
        self._strategy = strategy
        self._index = 0

    def select(self, provider: str) -> PooledCredential | None:
        """Select next available credential for provider."""
        available = [
            c
            for c in self._credentials
            if c.provider == provider and c.available
        ]
        if not available:
            return None

        if self._strategy == SelectionStrategy.ROUND_ROBIN:
            cred = available[self._index % len(available)]
            self._index += 1
        elif self._strategy == SelectionStrategy.LEAST_USED:
            cred = min(available, key=lambda c: c.use_count)
        else:  # FILL_FIRST
            cred = available[0]

        cred.use_count += 1
        return cred

    def report_error(
        self, credential: PooledCredential, error: ErrorInfo
    ) -> None:
        if error.status and (
            cooldown := COOLDOWN_POLICY.get(error.status)
        ):
            credential.cooldown_until = datetime.now() + cooldown

    def all_exhausted(self, provider: str) -> bool:
        return all(
            not c.available
            for c in self._credentials
            if c.provider == provider
        )


class CredentialExhaustedError(Exception):
    pass


async def create_pooled_provider(
    pool: CredentialPool,
    provider_name: str,
) -> Provider:
    """Create a provider using the next available credential from the pool."""
    cred = pool.select(provider_name)
    if cred is None:
        msg = f"No available credentials for {provider_name}"
        raise CredentialExhaustedError(msg)

    kwargs = {"api_key": cred.api_key}
    if cred.base_url:
        kwargs["base_url"] = cred.base_url
    if cred.extra:
        kwargs.update(cred.extra)

    return create_provider(provider_name, **kwargs)


# --- Context Probing ---


CONTEXT_PROBE_TIERS = [128_000, 64_000, 32_000, 16_000, 8_000]


async def probe_context_limit(provider: Provider, model: str) -> int:
    """Discover context limit by probing descending tiers."""
    for tier in CONTEXT_PROBE_TIERS:
        try:
            request = CompletionRequest(
                messages=[Message(role="user", content="hi")],
                model=model,
                max_tokens=1,
            )
            await provider.stream(request).result()
            return tier
        except Exception as e:
            if "context" in str(e).lower() or "token" in str(e).lower():
                continue
            raise
    return 8_000


# --- Known Model Registry ---


KNOWN_MODELS: dict[str, ModelInfo] = {
    "claude-sonnet-4-20250514": ModelInfo(
        id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        context_window=200_000,
        max_output_tokens=16_384,
        supports_tools=True,
        supports_vision=True,
        supports_thinking=True,
    ),
    "claude-opus-4-20250514": ModelInfo(
        id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        context_window=200_000,
        max_output_tokens=32_000,
        supports_tools=True,
        supports_vision=True,
        supports_thinking=True,
    ),
    "claude-haiku-4-5-20251001": ModelInfo(
        id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        context_window=200_000,
        max_output_tokens=8_192,
        supports_tools=True,
        supports_vision=True,
    ),
    "gpt-4o": ModelInfo(
        id="gpt-4o",
        display_name="GPT-4o",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_tools=True,
        supports_vision=True,
    ),
    "gpt-4o-mini": ModelInfo(
        id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_tools=True,
        supports_vision=True,
    ),
}


async def get_model_info(provider: Provider, model: str) -> ModelInfo:
    """Get model info from known registry, provider API, or probing."""
    if model in KNOWN_MODELS:
        return KNOWN_MODELS[model]

    try:
        models = await provider.list_models()
        for m in models:
            if m.id == model:
                return m
    except (NotImplementedError, Exception):
        pass

    ctx = await probe_context_limit(provider, model)
    return ModelInfo(
        id=model,
        display_name=model,
        context_window=ctx,
    )
