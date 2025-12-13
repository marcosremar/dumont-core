"""
Dumont Core - Módulos compartilhados para projetos Dumont

Módulos:
- llm: Gerenciador unificado de LLMs (OpenRouter, Ollama, OpenAI, Anthropic)
- cloud: Gestão de GPU na nuvem (Vast.ai, Skypilot)
- testing: Testes automatizados de UI com Browser-Use e LLMs
"""

__version__ = "0.1.0"

# Expose main modules using relative imports
from . import llm
from . import cloud
from . import testing

__all__ = ["llm", "cloud", "testing", "__version__"]
