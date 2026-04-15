from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from gatewayai.server.passthrough import (
    _UPSTREAMS,
    _build_upstream_headers,
    _resolve_credential,
    get_upstream,
    list_upstreams,
    passthrough,
    register_upstream,
)



def _make_request(
    method: str = "POST",
    meta: dict | None = None,
    body: bytes = b'{"model": "gpt-4o", "messages": []}',
) -> MagicMock:
    """Create a mock Django request."""
    request = MagicMock()
    request.method = method
    request.body = body
    request.META = {
        "HTTP_ACCEPT": "application/json",
        "HTTP_USER_AGENT": "test-client/1.0",
        "CONTENT_TYPE": "application/json",
        "QUERY_STRING": "",
    }
    if meta:
        request.META.update(meta)
    return request


class TestBuildUpstreamHeaders:
    def test_strips_authorization(self):
        request = _make_request(meta={"HTTP_AUTHORIZATION": "Bearer client-token"})
        upstream = {"base_url": "https://api.openai.com", "format": "openai"}
        headers = _build_upstream_headers(request, upstream, "sk-server-key")
        # Client auth stripped, server auth injected
        assert headers["authorization"] == "Bearer sk-server-key"

    def test_strips_x_api_key(self):
        request = _make_request(meta={"HTTP_X_API_KEY": "client-key"})
        upstream = {"base_url": "https://api.anthropic.com", "format": "anthropic"}
        headers = _build_upstream_headers(request, upstream, "sk-ant-key")
        assert headers["x-api-key"] == "sk-ant-key"
        assert "authorization" not in headers

    def test_strips_all_auth_variants(self):
        request = _make_request(
            meta={
                "HTTP_AUTHORIZATION": "Bearer x",
                "HTTP_X_API_KEY": "y",
                "HTTP_API_KEY": "z",
                "HTTP_X_GOOG_API_KEY": "w",
            }
        )
        upstream = {"base_url": "https://api.openai.com", "format": "openai"}
        headers = _build_upstream_headers(request, upstream, "sk-key")
        # Only the injected auth header should be present
        assert headers["authorization"] == "Bearer sk-key"
        assert "x-api-key" not in headers
        assert "api-key" not in headers
        assert "x-goog-api-key" not in headers

    def test_injects_openai_auth(self):
        request = _make_request()
        upstream = {"base_url": "https://api.openai.com", "format": "openai"}
        headers = _build_upstream_headers(request, upstream, "sk-123")
        assert headers["authorization"] == "Bearer sk-123"

    def test_injects_anthropic_auth(self):
        request = _make_request()
        upstream = {"base_url": "https://api.anthropic.com", "format": "anthropic"}
        headers = _build_upstream_headers(request, upstream, "sk-ant-456")
        assert headers["x-api-key"] == "sk-ant-456"

    def test_injects_google_auth(self):
        request = _make_request()
        upstream = {"base_url": "https://generativelanguage.googleapis.com", "format": "google"}
        headers = _build_upstream_headers(request, upstream, "AIza-key")
        assert headers["x-goog-api-key"] == "AIza-key"

    def test_preserves_content_type(self):
        request = _make_request()
        upstream = {"base_url": "https://api.openai.com", "format": "openai"}
        headers = _build_upstream_headers(request, upstream, "sk-123")
        assert headers["content-type"] == "application/json"

    def test_preserves_other_headers(self):
        request = _make_request(meta={"HTTP_X_CUSTOM_HEADER": "value"})
        upstream = {"base_url": "https://api.openai.com", "format": "openai"}
        headers = _build_upstream_headers(request, upstream, "sk-123")
        assert headers["x-custom-header"] == "value"
        assert headers["accept"] == "application/json"
        assert headers["user-agent"] == "test-client/1.0"

    def test_strips_host(self):
        request = _make_request(meta={"HTTP_HOST": "gateway.example.com"})
        upstream = {"base_url": "https://api.openai.com", "format": "openai"}
        headers = _build_upstream_headers(request, upstream, "sk-123")
        assert "host" not in headers


class TestUpstreamRegistry:
    def test_known_upstreams(self):
        assert get_upstream("openai") is not None
        assert get_upstream("anthropic") is not None
        assert get_upstream("google") is not None

    def test_unknown_upstream(self):
        assert get_upstream("nonexistent") is None

    def test_register_upstream(self):
        register_upstream("test-provider", "https://api.test.com")
        upstream = get_upstream("test-provider")
        assert upstream is not None
        assert upstream["base_url"] == "https://api.test.com"
        assert upstream["format"] == "openai"
        # Cleanup
        del _UPSTREAMS["test-provider"]

    def test_register_upstream_custom_format(self):
        register_upstream("test-anth", "https://custom.anthropic.com", format="anthropic")
        upstream = get_upstream("test-anth")
        assert upstream["format"] == "anthropic"
        del _UPSTREAMS["test-anth"]

    def test_register_upstream_invalid_format(self):
        with pytest.raises(ValueError, match="Unknown format"):
            register_upstream("bad", "https://example.com", format="invalid")

    def test_register_strips_trailing_slash(self):
        register_upstream("test-slash", "https://api.test.com/")
        assert get_upstream("test-slash")["base_url"] == "https://api.test.com"
        del _UPSTREAMS["test-slash"]

    def test_list_upstreams(self):
        names = list_upstreams()
        assert "openai" in names
        assert "anthropic" in names
        assert "google" in names


class TestResolveCredential:
    def test_single_key_from_header(self):
        request = _make_request(meta={"HTTP_X_PROVIDER_KEY": "sk-from-header"})
        cred = _resolve_credential("openai", request)
        assert cred is not None
        assert cred.api_key == "sk-from-header"

    def test_single_key_from_env(self):
        request = _make_request()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-from-env"}):
            cred = _resolve_credential("openai", request)
        assert cred is not None
        assert cred.api_key == "sk-from-env"

    def test_no_key_returns_none(self):
        request = _make_request()
        with patch.dict("os.environ", {}, clear=True):
            cred = _resolve_credential("openai", request)
        assert cred is None

    def test_pool_selection(self):
        from gatewayai.credentials import CredentialPool, PooledCredential

        pool = CredentialPool([
            PooledCredential(provider="openai", api_key="sk-1"),
            PooledCredential(provider="openai", api_key="sk-2"),
        ])
        request = _make_request()
        cred = _resolve_credential("openai", request, pool)
        assert cred is not None
        assert cred.api_key == "sk-1"

        cred2 = _resolve_credential("openai", request, pool)
        assert cred2.api_key == "sk-2"


class TestPassthroughView:
    @pytest.mark.asyncio
    async def test_unknown_provider_returns_404(self):
        request = _make_request()
        resp = await passthrough(request, "nonexistent", "v1/chat/completions")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_401(self):
        request = _make_request()
        with patch.dict("os.environ", {}, clear=True):
            resp = await passthrough(request, "openai", "v1/chat/completions")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_non_streaming_response(self):
        request = _make_request()
        upstream_body = b'{"id": "chatcmpl-123", "choices": []}'

        mock_resp = AsyncMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.headers = httpx.Headers({"content-type": "application/json"})
        mock_resp.aread = AsyncMock(return_value=upstream_body)
        mock_resp.aclose = AsyncMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_resp)

        with (
            patch("gatewayai.server.passthrough._get_client", return_value=mock_client),
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        ):
            resp = await passthrough(request, "openai", "v1/chat/completions")

        assert resp.status_code == 200
        assert resp.content == upstream_body

    @pytest.mark.asyncio
    async def test_streaming_sse_response(self):
        request = _make_request()

        chunks = [b"data: chunk1\n\n", b"data: chunk2\n\n", b"data: [DONE]\n\n"]

        mock_resp = AsyncMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.headers = httpx.Headers({"content-type": "text/event-stream"})
        mock_resp.aclose = AsyncMock()

        async def aiter_bytes(chunk_size=4096):
            for chunk in chunks:
                yield chunk

        mock_resp.aiter_bytes = aiter_bytes

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_resp)

        with (
            patch("gatewayai.server.passthrough._get_client", return_value=mock_client),
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        ):
            resp = await passthrough(request, "openai", "v1/chat/completions")

        assert resp.status_code == 200
        assert "text/event-stream" in resp["Content-Type"]

        # Collect streamed chunks
        collected = []
        async for chunk in resp.streaming_content:
            collected.append(chunk)
        assert collected == chunks

    @pytest.mark.asyncio
    async def test_upstream_connect_error_returns_502(self):
        request = _make_request()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with (
            patch("gatewayai.server.passthrough._get_client", return_value=mock_client),
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        ):
            resp = await passthrough(request, "openai", "v1/chat/completions")

        assert resp.status_code == 502

    @pytest.mark.asyncio
    async def test_upstream_timeout_returns_504(self):
        request = _make_request()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        with (
            patch("gatewayai.server.passthrough._get_client", return_value=mock_client),
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        ):
            resp = await passthrough(request, "openai", "v1/chat/completions")

        assert resp.status_code == 504

    @pytest.mark.asyncio
    async def test_preserves_upstream_status_code(self):
        """Upstream 422 is returned as-is to client."""
        request = _make_request()
        upstream_body = b'{"error": "invalid request"}'

        mock_resp = AsyncMock(spec=httpx.Response)
        mock_resp.status_code = 422
        mock_resp.headers = httpx.Headers({"content-type": "application/json"})
        mock_resp.aread = AsyncMock(return_value=upstream_body)
        mock_resp.aclose = AsyncMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_resp)

        with (
            patch("gatewayai.server.passthrough._get_client", return_value=mock_client),
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        ):
            resp = await passthrough(request, "openai", "v1/chat/completions")

        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_propagates_response_headers(self):
        request = _make_request()

        mock_resp = AsyncMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.headers = httpx.Headers({
            "content-type": "application/json",
            "x-request-id": "req-abc",
            "x-ratelimit-remaining": "99",
        })
        mock_resp.aread = AsyncMock(return_value=b'{}')
        mock_resp.aclose = AsyncMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_resp)

        with (
            patch("gatewayai.server.passthrough._get_client", return_value=mock_client),
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        ):
            resp = await passthrough(request, "openai", "v1/chat/completions")

        assert resp["x-request-id"] == "req-abc"
        assert resp["x-ratelimit-remaining"] == "99"

    @pytest.mark.asyncio
    async def test_url_construction_with_query_string(self):
        request = _make_request(meta={"QUERY_STRING": "limit=10&offset=0"})

        mock_resp = AsyncMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.headers = httpx.Headers({"content-type": "application/json"})
        mock_resp.aread = AsyncMock(return_value=b'[]')
        mock_resp.aclose = AsyncMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_resp)

        with (
            patch("gatewayai.server.passthrough._get_client", return_value=mock_client),
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        ):
            await passthrough(request, "openai", "v1/models")

        # Verify the URL includes query string
        call_args = mock_client.build_request.call_args
        assert call_args[0][1] == "https://api.openai.com/v1/models?limit=10&offset=0"


class TestPassthroughCredentialPool:
    @pytest.mark.asyncio
    async def test_rotates_keys_on_429(self):
        from gatewayai.credentials import CredentialPool, PooledCredential

        pool = CredentialPool([
            PooledCredential(provider="openai", api_key="sk-1"),
            PooledCredential(provider="openai", api_key="sk-2"),
        ])
        request = _make_request()

        # First call returns 429, second succeeds
        mock_resp_429 = AsyncMock(spec=httpx.Response)
        mock_resp_429.status_code = 429
        mock_resp_429.headers = httpx.Headers({"content-type": "application/json"})
        mock_resp_429.aclose = AsyncMock()

        mock_resp_200 = AsyncMock(spec=httpx.Response)
        mock_resp_200.status_code = 200
        mock_resp_200.headers = httpx.Headers({"content-type": "application/json"})
        mock_resp_200.aread = AsyncMock(return_value=b'{"ok": true}')
        mock_resp_200.aclose = AsyncMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(side_effect=[mock_resp_429, mock_resp_200])

        with (
            patch("gatewayai.server.passthrough._get_client", return_value=mock_client),
            patch("gatewayai.server.passthrough._pool", pool),
        ):
            resp = await passthrough(request, "openai", "v1/chat/completions")

        assert resp.status_code == 200
        # First key should be cooled down
        assert not pool._credentials[0].available

    @pytest.mark.asyncio
    async def test_all_keys_exhausted_returns_429(self):
        from datetime import datetime, timedelta

        from gatewayai.credentials import CredentialPool, PooledCredential

        pool = CredentialPool([
            PooledCredential(
                provider="openai",
                api_key="sk-1",
                cooldown_until=datetime.now() + timedelta(hours=1),
            ),
        ])
        request = _make_request()

        with patch("gatewayai.server.passthrough._pool", pool):
            resp = await passthrough(request, "openai", "v1/chat/completions")

        assert resp.status_code == 429

    @pytest.mark.asyncio
    async def test_single_key_fallback_no_pool(self):
        """Without a pool, uses _get_api_key() for a single credential."""
        request = _make_request()

        mock_resp = AsyncMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.headers = httpx.Headers({"content-type": "application/json"})
        mock_resp.aread = AsyncMock(return_value=b'{}')
        mock_resp.aclose = AsyncMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_resp)

        with (
            patch("gatewayai.server.passthrough._get_client", return_value=mock_client),
            patch("gatewayai.server.passthrough._pool", None),
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env-key"}),
        ):
            resp = await passthrough(request, "openai", "v1/chat/completions")

        assert resp.status_code == 200
