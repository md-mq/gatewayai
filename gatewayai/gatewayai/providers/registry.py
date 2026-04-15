from __future__ import annotations

from typing import Any, Callable

from gatewayai.providers.protocol import Provider

_PROVIDERS: dict[str, Callable[..., Provider]] = {}


def register_provider(name: str, factory: Callable[..., Provider]) -> None:
    _PROVIDERS[name] = factory


def create_provider(name: str, **kwargs: Any) -> Provider:
    """Create a provider by name. Lazy-imports the provider module."""
    if name not in _PROVIDERS:
        _lazy_register(name)
    factory = _PROVIDERS[name]
    return factory(**kwargs)


def _lazy_register(name: str) -> None:
    """Import and register a provider on first use."""
    match name:
        case "anthropic":
            from gatewayai.providers.anthropic import AnthropicProvider

            register_provider("anthropic", AnthropicProvider)
        case "openai":
            from gatewayai.providers.openai import OpenAIProvider

            register_provider("openai", OpenAIProvider)
        case "google":
            from gatewayai.providers.google import GoogleProvider

            register_provider("google", GoogleProvider)
        case "ollama":
            from gatewayai.providers.ollama import OllamaProvider

            register_provider("ollama", OllamaProvider)
        case "deepseek":
            from gatewayai.providers.deepseek import DeepSeekProvider

            register_provider("deepseek", DeepSeekProvider)
        case "bedrock":
            from gatewayai.providers.bedrock import BedrockAnthropicProvider

            register_provider("bedrock", BedrockAnthropicProvider)
        case "vertex":
            from gatewayai.providers.vertex import VertexAnthropicProvider

            register_provider("vertex", VertexAnthropicProvider)
        case "azure":
            from gatewayai.providers.azure import AzureOpenAIProvider

            register_provider("azure", AzureOpenAIProvider)
        case _:
            msg = f"Unknown provider: {name}"
            raise ValueError(msg)


def list_providers() -> list[str]:
    return list(_PROVIDERS.keys())
