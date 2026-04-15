from __future__ import annotations

from typing import Any

from gatewayai.providers.openai import OpenAIProvider


class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI provider via openai.AsyncAzureOpenAI."""

    def __init__(
        self,
        azure_endpoint: str,
        api_version: str = "2024-10-21",
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        **kwargs: Any,
    ):
        import openai

        self._client = openai.AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            **kwargs,
        )

    @property
    def id(self) -> str:
        return "azure"
