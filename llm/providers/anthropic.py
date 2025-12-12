"""Anthropic provider strategy."""

from __future__ import annotations

import os

from .base import LLMProviderStrategy


class AnthropicProvider(LLMProviderStrategy):
    """Strategy for Anthropic provider."""

    def configure(self, model: str, api_key: str) -> None:
        """
        Configure Anthropic provider.

        Sets ANTHROPIC_API_KEY environment variable for litellm.
        """
        os.environ["ANTHROPIC_API_KEY"] = api_key

    def get_model_name(self, model: str) -> str:
        """
        Get formatted model name for Anthropic.

        Args:
            model: Original model name

        Returns:
            Model name as-is (Anthropic doesn't need prefix for litellm)
        """
        return model

    def supports_function_calling(self, model: str) -> bool:
        """
        Check if model supports function calling.

        Claude models support function calling.
        """
        return "claude" in model.lower()
