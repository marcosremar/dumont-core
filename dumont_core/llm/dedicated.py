#!/usr/bin/env python3
"""
Dedicated Machine Provider - Provisiona máquinas GPU via Vast.ai

Suporta dois backends:
- Ollama: Para modelos do Ollama Registry
- HuggingFace/vLLM: Para modelos do HuggingFace Hub
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from dumont_core.llm.tunnel import SSHTunnel, RemoteServerConfig

logger = logging.getLogger("dumont_core.llm.dedicated")


class DedicatedBackend(Enum):
    """Backends suportados para máquinas dedicadas"""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"  # Alias para huggingface


@dataclass
class DedicatedConfig:
    """Configuração para máquina dedicada"""
    backend: DedicatedBackend = DedicatedBackend.OLLAMA
    model: str = "qwen2.5-coder:14b"
    
    # Vast.ai config
    max_cost_per_hour: float = 0.50
    gpu_type: str = "RTX 3090"
    min_gpu_memory_gb: int = 24
    
    # HuggingFace specific
    quantization: Optional[str] = None  # awq, gptq, None
    
    # Timeouts
    provision_timeout_minutes: int = 10
    model_load_timeout_minutes: int = 15
    
    # Local port for tunnel
    local_port: int = 11435


@dataclass 
class DedicatedInstance:
    """Representa uma instância dedicada provisionada"""
    instance_id: int
    backend: DedicatedBackend
    model: str
    ssh_host: str
    ssh_port: int
    local_port: int
    api_port: int = 11434  # Ollama default
    
    _tunnel: Optional[SSHTunnel] = field(default=None, repr=False)
    
    @property
    def endpoint(self) -> str:
        """Endpoint local para acessar o modelo"""
        return f"http://localhost:{self.local_port}"
    
    async def connect(self) -> bool:
        """Estabelece túnel SSH para a instância"""
        if self._tunnel and self._tunnel.is_active():
            return True
        
        config = RemoteServerConfig(
            host=self.ssh_host,
            ssh_port=self.ssh_port,
            ollama_port=self.api_port,
            local_port=self.local_port,
            user="root",
        )
        
        self._tunnel = SSHTunnel(config)
        return self._tunnel.start()
    
    def disconnect(self):
        """Fecha o túnel SSH"""
        if self._tunnel:
            self._tunnel.stop()
            self._tunnel = None


class DedicatedProvider:
    """
    Provider para máquinas dedicadas via Vast.ai.
    
    Provisiona máquinas GPU e instala o backend necessário (Ollama ou vLLM).
    
    Uso:
        provider = DedicatedProvider()
        instance = await provider.provision(
            backend="ollama",
            model="qwen3-coder:30b"
        )
        
        # Conectar via túnel SSH
        await instance.connect()
        
        # Usar endpoint local
        llm = ChatOllama(base_url=instance.endpoint, model=model)
    """
    
    def __init__(self):
        self._instances: dict[int, DedicatedInstance] = {}
        self._vastai_api_key = os.environ.get("VASTAI_API_KEY")
    
    @property
    def is_available(self) -> bool:
        """Verifica se Vast.ai está configurado"""
        return bool(self._vastai_api_key)
    
    async def provision(
        self,
        model: str,
        backend: str = "ollama",
        config: Optional[DedicatedConfig] = None,
    ) -> DedicatedInstance:
        """
        Provisiona uma nova máquina dedicada.
        
        Args:
            model: Nome do modelo (Ollama format ou HuggingFace repo)
            backend: "ollama" ou "huggingface"
            config: Configurações adicionais
        
        Returns:
            DedicatedInstance pronta para uso
        """
        if not self.is_available:
            raise RuntimeError("VASTAI_API_KEY não configurada")
        
        config = config or DedicatedConfig()
        backend_enum = DedicatedBackend(backend)
        
        logger.info(f"Provisionando máquina dedicada: {backend}/{model}")
        
        # 1. Buscar melhor oferta no Vast.ai
        offer = await self._find_best_offer(config)
        if not offer:
            raise RuntimeError("Nenhuma máquina disponível no Vast.ai")
        
        # 2. Criar instância
        instance_id = await self._create_instance(offer, backend_enum, model, config)
        
        # 3. Aguardar instância ficar pronta
        ssh_info = await self._wait_for_instance(instance_id, config.provision_timeout_minutes)
        
        # 4. Instalar backend e modelo
        await self._setup_instance(instance_id, backend_enum, model, config)
        
        # 5. Criar objeto DedicatedInstance
        instance = DedicatedInstance(
            instance_id=instance_id,
            backend=backend_enum,
            model=model,
            ssh_host=ssh_info["host"],
            ssh_port=ssh_info["port"],
            local_port=config.local_port,
            api_port=11434 if backend_enum == DedicatedBackend.OLLAMA else 8000,
        )
        
        self._instances[instance_id] = instance
        
        # 6. Estabelecer túnel
        if not await instance.connect():
            raise RuntimeError("Falha ao estabelecer túnel SSH")
        
        logger.info(f"Máquina dedicada pronta: {instance.endpoint}")
        return instance
    
    async def _find_best_offer(self, config: DedicatedConfig) -> Optional[dict]:
        """Busca a melhor oferta no Vast.ai"""
        try:
            from dumont_core.cloud.gpu_lifecycle import GPUInstanceState
            # TODO: Implementar busca real via Vast.ai API
            # Por enquanto, usar a infraestrutura existente do cloud module
            logger.info(f"Buscando GPU {config.gpu_type} <= ${config.max_cost_per_hour}/hr")
            return {"id": "mock", "price": 0.30}
        except ImportError:
            logger.warning("Cloud module não disponível para busca de ofertas")
            return None
    
    async def _create_instance(
        self, 
        offer: dict, 
        backend: DedicatedBackend,
        model: str,
        config: DedicatedConfig
    ) -> int:
        """Cria instância no Vast.ai"""
        # TODO: Implementar criação real via Vast.ai API
        logger.info(f"Criando instância para {backend.value}/{model}")
        return 12345  # Mock instance ID
    
    async def _wait_for_instance(
        self, 
        instance_id: int, 
        timeout_minutes: int
    ) -> dict:
        """Aguarda instância ficar pronta"""
        logger.info(f"Aguardando instância {instance_id} ficar pronta...")
        # TODO: Implementar polling real
        await asyncio.sleep(2)  # Mock delay
        return {"host": "example.com", "port": 22}
    
    async def _setup_instance(
        self,
        instance_id: int,
        backend: DedicatedBackend,
        model: str,
        config: DedicatedConfig
    ):
        """Instala backend e modelo na instância"""
        if backend == DedicatedBackend.OLLAMA:
            await self._setup_ollama(instance_id, model)
        else:
            await self._setup_vllm(instance_id, model, config.quantization)
    
    async def _setup_ollama(self, instance_id: int, model: str):
        """Instala Ollama e baixa modelo"""
        logger.info(f"Instalando Ollama e baixando {model}...")
        # TODO: Executar via SSH
        # curl -fsSL https://ollama.com/install.sh | sh
        # ollama pull {model}
        await asyncio.sleep(1)
    
    async def _setup_vllm(
        self, 
        instance_id: int, 
        model: str,
        quantization: Optional[str]
    ):
        """Instala vLLM e baixa modelo do HuggingFace"""
        logger.info(f"Instalando vLLM e baixando {model}...")
        # TODO: Executar via SSH
        # pip install vllm
        # python -m vllm.entrypoints.openai.api_server --model {model}
        await asyncio.sleep(1)
    
    async def get_or_create(
        self,
        model: str,
        backend: str = "ollama",
        **kwargs
    ) -> DedicatedInstance:
        """
        Obtém instância existente ou cria nova.
        
        Reutiliza instâncias já provisionadas se o modelo for o mesmo.
        """
        # Verificar instâncias existentes
        for instance in self._instances.values():
            if instance.model == model and instance.backend.value == backend:
                if await instance.connect():
                    logger.info(f"Reutilizando instância existente: {instance.instance_id}")
                    return instance
        
        # Criar nova
        return await self.provision(model, backend, **kwargs)
    
    def list_instances(self) -> list[DedicatedInstance]:
        """Lista instâncias ativas"""
        return list(self._instances.values())
    
    async def terminate(self, instance_id: int):
        """Termina uma instância"""
        if instance_id in self._instances:
            instance = self._instances[instance_id]
            instance.disconnect()
            del self._instances[instance_id]
            logger.info(f"Instância {instance_id} terminada")
            # TODO: Terminar no Vast.ai
    
    async def terminate_all(self):
        """Termina todas as instâncias"""
        for instance_id in list(self._instances.keys()):
            await self.terminate(instance_id)


# Singleton
_provider: Optional[DedicatedProvider] = None


def get_dedicated_provider() -> DedicatedProvider:
    """Obtém instância singleton do DedicatedProvider"""
    global _provider
    if _provider is None:
        _provider = DedicatedProvider()
    return _provider
