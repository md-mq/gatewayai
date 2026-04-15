from __future__ import annotations

from gatewayai.providers.openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek provider — OpenAI-compatible API."""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key, base_url="https://api.deepseek.com/v1"
        )

    @property
    def id(self) -> str:
        return "deepseek"
