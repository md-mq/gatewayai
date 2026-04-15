import pytest

from gatewayai.providers.registry import (
    _PROVIDERS,
    create_provider,
    list_providers,
    register_provider,
)


class TestRegistry:
    def setup_method(self):
        # Clear registry between tests
        _PROVIDERS.clear()

    def test_register_and_create(self):
        class FakeProvider:
            def __init__(self, api_key: str):
                self.api_key = api_key

            @property
            def id(self):
                return "fake"

        register_provider("fake", FakeProvider)
        provider = create_provider("fake", api_key="sk-test")
        assert provider.id == "fake"
        assert provider.api_key == "sk-test"

    def test_list_providers(self):
        register_provider("a", lambda: None)
        register_provider("b", lambda: None)
        assert set(list_providers()) == {"a", "b"}

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider: nonexistent"):
            create_provider("nonexistent")

    def test_lazy_register_anthropic(self):
        # This will attempt to import anthropic, which may not be installed
        # in test env. Just verify the registry mechanism.
        try:
            provider = create_provider("anthropic", api_key="sk-test")
            assert provider.id == "anthropic"
        except ImportError:
            pytest.skip("anthropic SDK not installed")

    def test_lazy_register_openai(self):
        try:
            provider = create_provider("openai", api_key="sk-test")
            assert provider.id == "openai"
        except ImportError:
            pytest.skip("openai SDK not installed")
