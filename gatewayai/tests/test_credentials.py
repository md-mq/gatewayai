from datetime import datetime, timedelta

import pytest

from gatewayai.credentials import (
    COOLDOWN_POLICY,
    CredentialPool,
    PooledCredential,
    SelectionStrategy,
)
from gatewayai.types import ErrorCode, ErrorInfo


class TestPooledCredential:
    def test_available_by_default(self):
        cred = PooledCredential(provider="anthropic", api_key="sk-123")
        assert cred.available

    def test_not_available_during_cooldown(self):
        cred = PooledCredential(
            provider="anthropic",
            api_key="sk-123",
            cooldown_until=datetime.now() + timedelta(hours=1),
        )
        assert not cred.available

    def test_available_after_cooldown(self):
        cred = PooledCredential(
            provider="anthropic",
            api_key="sk-123",
            cooldown_until=datetime.now() - timedelta(seconds=1),
        )
        assert cred.available


class TestCredentialPool:
    def setup_method(self):
        self.creds = [
            PooledCredential(provider="anthropic", api_key="sk-1"),
            PooledCredential(provider="anthropic", api_key="sk-2"),
            PooledCredential(provider="openai", api_key="sk-oai"),
        ]

    def test_round_robin(self):
        pool = CredentialPool(
            self.creds, strategy=SelectionStrategy.ROUND_ROBIN
        )
        c1 = pool.select("anthropic")
        c2 = pool.select("anthropic")
        assert c1.api_key == "sk-1"
        assert c2.api_key == "sk-2"

    def test_least_used(self):
        pool = CredentialPool(
            self.creds, strategy=SelectionStrategy.LEAST_USED
        )
        self.creds[0].use_count = 5
        cred = pool.select("anthropic")
        assert cred.api_key == "sk-2"

    def test_fill_first(self):
        pool = CredentialPool(
            self.creds, strategy=SelectionStrategy.FILL_FIRST
        )
        c1 = pool.select("anthropic")
        c2 = pool.select("anthropic")
        assert c1.api_key == "sk-1"
        assert c2.api_key == "sk-1"

    def test_select_none_for_unknown_provider(self):
        pool = CredentialPool(self.creds)
        assert pool.select("gemini") is None

    def test_skips_cooled_down(self):
        self.creds[0].cooldown_until = datetime.now() + timedelta(hours=1)
        pool = CredentialPool(
            self.creds, strategy=SelectionStrategy.ROUND_ROBIN
        )
        cred = pool.select("anthropic")
        assert cred.api_key == "sk-2"

    def test_all_exhausted(self):
        for c in self.creds:
            if c.provider == "anthropic":
                c.cooldown_until = datetime.now() + timedelta(hours=1)
        pool = CredentialPool(self.creds)
        assert pool.all_exhausted("anthropic")
        assert not pool.all_exhausted("openai")

    def test_report_error_applies_cooldown(self):
        pool = CredentialPool(self.creds)
        cred = self.creds[0]
        error = ErrorInfo(
            code=ErrorCode.RATE_LIMIT, message="rate limited", status=429
        )
        pool.report_error(cred, error)
        assert cred.cooldown_until is not None
        assert not cred.available

    def test_report_error_no_cooldown_for_unknown_status(self):
        pool = CredentialPool(self.creds)
        cred = self.creds[0]
        error = ErrorInfo(
            code=ErrorCode.UNKNOWN, message="something", status=400
        )
        pool.report_error(cred, error)
        assert cred.cooldown_until is None
