from __future__ import annotations

from typing import Any

from gatewayai.providers.anthropic import AnthropicProvider


class BedrockAnthropicProvider(AnthropicProvider):
    """Claude on AWS Bedrock via anthropic[bedrock] SDK."""

    def __init__(
        self,
        aws_access_key: str | None = None,
        aws_secret_key: str | None = None,
        aws_region: str | None = None,
        aws_session_token: str | None = None,
        **kwargs: Any,
    ):
        import anthropic

        self._client = anthropic.AsyncAnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region or "us-east-1",
            aws_session_token=aws_session_token,
            **kwargs,
        )

    @property
    def id(self) -> str:
        return "bedrock"
