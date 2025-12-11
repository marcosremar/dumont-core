"""
Dumont Core - Módulos compartilhados para projetos Dumont

Módulos:
- llm: Gerenciador unificado de LLMs (OpenRouter, Ollama, OpenAI, Anthropic)
- cloud: Gestão de GPU na nuvem (Vast.ai, Skypilot)
"""

__version__ = "0.1.0"

# Expose main modules
from dumont_core import llm
from dumont_core import cloud

__all__ = ["llm", "cloud", "__version__"]
