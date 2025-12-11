"""Factory for creating LLM provider strategies."""

from __future__ import annotations

from dumont_core.infrastructure.llm_providers.base import LLMProviderStrategy
from dumont_core.infrastructure.llm_providers.openai import OpenAIProvider
from dumont_core.infrastructure.llm_providers.openrouter import OpenRouterProvider
from dumont_core.infrastructure.llm_providers.anthropic import AnthropicProvider
from dumont_core.infrastructure.llm_providers.ollama import OllamaProvider


def create_provider(provider_name: str) -> LLMProviderStrategy:
    """
    Create an LLM provider strategy based on provider name.

    Args:
        provider_name: Provider name ("openai", "openrouter", "ollama", etc.)

    Returns:
        Provider strategy instance

    Raises:
        ValueError: If provider is not supported
    """
    provider_name_lower = provider_name.lower()

    if provider_name_lower == "openrouter":
        return OpenRouterProvider()
    elif provider_name_lower == "openai":
        return OpenAIProvider()
    elif provider_name_lower == "anthropic":
        return AnthropicProvider()
    elif provider_name_lower in ("ollama", "local"):
        return OllamaProvider()
    else:
        raise ValueError(
            f"Unsupported provider: {provider_name}. "
            f"Supported providers: openai, openrouter, anthropic, ollama"
        )

