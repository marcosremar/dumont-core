"""
Dumont Core LLM - Gerenciador unificado de LLMs

Suporta múltiplos providers:
- OpenRouter (API cloud via LiteLLM)
- Ollama (local e remoto via SSH)
- Dedicated (máquina dedicada via Vast.ai)
- OpenAI, Anthropic

Features:
- Auto-seleção de provider
- Túnel SSH automático para VPS
- Provisionamento de máquinas GPU (Vast.ai)
- Gateway com litellm
- Fallback entre providers
"""

from .manager import (
    LLMManager,
    LLMProvider,
    LLMConfig,
    get_llm_manager,
    get_llm,
    list_models,
)

from .tunnel import SSHTunnel, RemoteServerConfig

# Dedicated machine provider
from .dedicated import (
    DedicatedProvider,
    DedicatedInstance,
    DedicatedConfig,
    DedicatedBackend,
    get_dedicated_provider,
)

# Gateway avançado (opcional - requer litellm)
try:
    from .gateway import LLMGateway, ChatResponse, ToolCall
    GATEWAY_AVAILABLE = True
except ImportError:
    GATEWAY_AVAILABLE = False
    LLMGateway = None
    ChatResponse = None
    ToolCall = None

__all__ = [
    # Manager simples
    "LLMManager",
    "LLMProvider", 
    "LLMConfig",
    "get_llm_manager",
    "get_llm",
    "list_models",
    # SSH Tunnel
    "SSHTunnel",
    "RemoteServerConfig",
    # Dedicated machines
    "DedicatedProvider",
    "DedicatedInstance",
    "DedicatedConfig",
    "DedicatedBackend",
    "get_dedicated_provider",
    # Gateway avançado
    "LLMGateway",
    "ChatResponse",
    "ToolCall",
    "GATEWAY_AVAILABLE",
]
