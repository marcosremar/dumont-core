"""OpenAI provider strategy."""

from __future__ import annotations

import litellm

from .base import LLMProviderStrategy


class OpenAIProvider(LLMProviderStrategy):
    """Strategy for OpenAI provider."""

    def configure(self, model: str, api_key: str) -> None:
        """
        Configure OpenAI provider.

        Sets litellm API key.
        """
        litellm.api_key = api_key

    def get_model_name(self, model: str) -> str:
        """
        Get formatted model name for OpenAI.

        Args:
            model: Original model name

        Returns:
            Model name as-is (OpenAI doesn't need prefix)
        """
        return model

    def supports_function_calling(self, model: str) -> bool:
        """
        Check if model supports function calling.

        OpenAI GPT models support function calling.
        """
        supported_prefixes = ["gpt-", "claude-", "gemini-"]
        return any(model.startswith(prefix) for prefix in supported_prefixes)

