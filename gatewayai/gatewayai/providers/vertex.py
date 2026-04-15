from __future__ import annotations

from typing import Any

from gatewayai.providers.anthropic import AnthropicProvider


class VertexAnthropicProvider(AnthropicProvider):
    """Claude on Google Vertex AI via anthropic[vertex] SDK."""

    def __init__(
        self,
        project_id: str | None = None,
        region: str | None = None,
        **kwargs: Any,
    ):
        import anthropic

        self._client = anthropic.AsyncAnthropicVertex(
            project_id=project_id,
            region=region or "us-east5",
            **kwargs,
        )

    @property
    def id(self) -> str:
        return "vertex"
