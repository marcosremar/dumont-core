"""OpenRouter provider strategy."""

from __future__ import annotations

import os

from dumont_core.infrastructure.llm_providers.base import LLMProviderStrategy


class OpenRouterProvider(LLMProviderStrategy):
    """Strategy for OpenRouter provider."""

    def configure(self, model: str, api_key: str) -> None:
        """
        Configure OpenRouter provider.

        Sets OPENROUTER_API_KEY environment variable.
        """
        os.environ["OPENROUTER_API_KEY"] = api_key

    def get_model_name(self, model: str) -> str:
        """
        Get formatted model name for OpenRouter.

        Args:
            model: Original model name

        Returns:
            Model name formatted for LiteLLM with openrouter/ prefix
        """
        # If already has openrouter/ prefix, return as is
        if model.startswith("openrouter/"):
            return model

        # If model already has provider prefix (e.g., openai/gpt-4o-mini)
        # add openrouter/ prefix
        if "/" in model:
            return f"openrouter/{model}"

        # For simple model names (e.g., gpt-4o-mini), assume OpenAI as provider
        # LiteLLM requires openrouter/provider/model format
        return f"openrouter/openai/{model}"

    # Models that are known NOT to support function calling via OpenRouter
    # These will use XML tool calling instead
    MODELS_WITHOUT_FUNCTION_CALLING = {
        # Qwen models - no tool use support via OpenRouter
        "qwen/qwen-2.5-coder-32b-instruct",
        "qwen/qwen-2.5-72b-instruct",
        "qwen/qwen-2.5-7b-instruct",
        "qwen/qwen-2-72b-instruct",
        "qwen/qwen-2-7b-instruct",
        "qwen/qwq-32b-preview",
        # DeepSeek models
        "deepseek/deepseek-chat",
        "deepseek/deepseek-coder",
        "deepseek/deepseek-r1",
        # Google models (use their own API for function calling)
        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it",
        # Mixtral (some variants)
        "mistralai/mixtral-8x7b-instruct",
        # Meta Llama (some variants don't have tool support)
        "meta-llama/llama-3.2-1b-instruct",
        "meta-llama/llama-3.2-3b-instruct",
    }

    def supports_function_calling(self, model: str) -> bool:
        """
        Check if model supports function calling.

        Most OpenRouter models support function calling, but some
        (like Qwen, DeepSeek) don't have tool use endpoints.
        """
        formatted_model = self.get_model_name(model)

        # Remove openrouter/ prefix for checking
        clean_model = formatted_model.replace("openrouter/", "")

        # Check if model is in the exclusion list
        if clean_model in self.MODELS_WITHOUT_FUNCTION_CALLING:
            return False

        # Also check partial matches (for model variants)
        for excluded in self.MODELS_WITHOUT_FUNCTION_CALLING:
            if excluded in clean_model or clean_model in excluded:
                return False

        # Default: assume function calling is supported
        return True

