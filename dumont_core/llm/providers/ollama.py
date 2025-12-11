"""Ollama LLM provider strategy."""

from __future__ import annotations

import os

from dumont_core.infrastructure.llm_providers.base import LLMProviderStrategy


class OllamaProvider(LLMProviderStrategy):
    """
    Ollama LLM provider strategy.

    For LiteLLM, Ollama models need the "ollama/" prefix.
    Ollama runs locally so API key is not strictly required.
    """

    def configure(self, model: str, api_key: str) -> None:
        """Configure Ollama provider.

        Args:
            model: Model name (e.g., "qwen2.5-coder:7b-instruct")
            api_key: API key (not used for local Ollama, can be "ollama" or empty)
        """
        self.model = model
        # Ollama API base URL (default is localhost:11434)
        self.api_base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")

    def get_model_name(self, model: str) -> str:
        """Get LiteLLM-formatted model name for Ollama.

        Args:
            model: Original model name

        Returns:
            Model name with "ollama/" prefix for LiteLLM
        """
        # LiteLLM requires "ollama/" prefix for Ollama models
        if model.startswith("ollama/"):
            return model
        return f"ollama/{model}"

    def supports_function_calling(self, model: str) -> bool:
        """Check if Ollama model supports function calling.

        Most modern Ollama models support function calling/tool use.

        Args:
            model: Model name

        Returns:
            True - assume function calling is supported
        """
        # Most modern Ollama models (Qwen, Llama, etc.) support function calling
        return True
