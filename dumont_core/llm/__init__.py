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

from dumont_core.llm.manager import (
    LLMManager,
    LLMProvider,
    LLMConfig,
    get_llm_manager,
    get_llm,
    list_models,
)

from dumont_core.llm.tunnel import SSHTunnel, RemoteServerConfig

# Dedicated machine provider
from dumont_core.llm.dedicated import (
    DedicatedProvider,
    DedicatedInstance,
    DedicatedConfig,
    DedicatedBackend,
    get_dedicated_provider,
)

# Gateway avançado (opcional - requer litellm)
try:
    from dumont_core.llm.gateway import LLMGateway, ChatResponse, ToolCall
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
