"""
Providers - Provedores de LLM específicos
"""

from .openrouter import OpenRouterProvider
from .ollama import OllamaProvider

__all__ = ["OpenRouterProvider", "OllamaProvider"]
