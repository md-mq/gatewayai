from __future__ import annotations

from gatewayai.providers.openai import OpenAIProvider


class OllamaProvider(OpenAIProvider):
    """Ollama provider — OpenAI-compatible local inference."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ):
        super().__init__(api_key=api_key, base_url=base_url)

    @property
    def id(self) -> str:
        return "ollama"
