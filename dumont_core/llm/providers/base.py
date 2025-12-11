"""Base class for LLM provider strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProviderStrategy(ABC):
    """
    Abstract base class for LLM provider strategies.

    Each provider implements this interface to handle provider-specific
    configuration and model name formatting.
    """

    @abstractmethod
    def configure(self, model: str, api_key: str) -> None:
        """
        Configure the provider with API key and model.

        Args:
            model: Model name
            api_key: API key for the provider
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_name(self, model: str) -> str:
        """
        Get the formatted model name for this provider.

        Args:
            model: Original model name

        Returns:
            Formatted model name for the provider
        """
        raise NotImplementedError

    @abstractmethod
    def supports_function_calling(self, model: str) -> bool:
        """
        Check if the model supports function calling.

        Args:
            model: Model name

        Returns:
            True if function calling is supported
        """
        raise NotImplementedError

