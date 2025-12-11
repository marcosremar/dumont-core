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
import subprocess
import json
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
    gpu_type: str = "RTX_3090"  # RTX_3090, RTX_4090, A100, etc.
    min_gpu_memory_gb: int = 24
    min_disk_gb: int = 50
    
    # Docker image
    docker_image: str = "ollama/ollama:latest"
    
    # HuggingFace specific
    quantization: Optional[str] = None  # awq, gptq, None
    
    # Timeouts
    provision_timeout_seconds: int = 300
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
    
    async def _run_vastai_command(self, args: list[str]) -> subprocess.CompletedProcess:
        """Executa comando vastai CLI"""
        vastai_paths = [
            "vastai",
            os.path.expanduser("~/.local/bin/vastai"),
            "/home/marcos/.local/bin/vastai",  # Common Ubuntu location
            "/usr/local/bin/vastai",
        ]
        
        vastai_cmd = "vastai"
        for path in vastai_paths:
            if os.path.exists(path):
                vastai_cmd = path
                break
        
        cmd = [vastai_cmd, "--api-key", self._vastai_api_key] + args
        logger.debug(f"Running: vastai {' '.join(args)}")
        
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True
        )
        return result
    
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
            raise RuntimeError("Nenhuma máquina disponível no Vast.ai com os requisitos especificados")
        
        logger.info(f"Melhor oferta: GPU {offer.get('gpu_name', '?')} @ ${offer.get('dph_total', '?')}/hr")
        
        # 2. Criar instância
        instance_id = await self._create_instance(offer, backend_enum, model, config)
        
        # 3. Aguardar instância ficar pronta
        instance_info = await self._wait_for_instance(instance_id, config.provision_timeout_seconds)
        
        # 4. Criar objeto DedicatedInstance
        instance = DedicatedInstance(
            instance_id=instance_id,
            backend=backend_enum,
            model=model,
            ssh_host=instance_info["host"],
            ssh_port=instance_info["port"],
            local_port=config.local_port,
            api_port=11434 if backend_enum == DedicatedBackend.OLLAMA else 8000,
        )
        
        self._instances[instance_id] = instance
        
        # 5. Estabelecer túnel
        if not await instance.connect():
            raise RuntimeError("Falha ao estabelecer túnel SSH")
        
        # 6. Aguardar modelo carregar (Ollama puxa automaticamente no onstart)
        if backend_enum == DedicatedBackend.OLLAMA:
            await self._wait_for_ollama_ready(instance, model, config.model_load_timeout_minutes)
        
        logger.info(f"Máquina dedicada pronta: {instance.endpoint}")
        return instance
    
    async def _find_best_offer(self, config: DedicatedConfig) -> Optional[dict]:
        """Busca a melhor oferta no Vast.ai"""
        # Construir query de busca
        query = (
            f"gpu_name={config.gpu_type} "
            f"gpu_ram>={config.min_gpu_memory_gb} "
            f"disk_space>={config.min_disk_gb} "
            f"dph_total<={config.max_cost_per_hour} "
            f"reliability>0.9 "
            f"rentable=true"
        )
        
        result = await self._run_vastai_command([
            "search", "offers", query,
            "--order", "dph_total",
            "--limit", "1",
            "--raw"
        ])
        
        if result.returncode != 0:
            logger.error(f"Falha ao buscar ofertas: {result.stderr}")
            return None
        
        try:
            offers = json.loads(result.stdout)
            if offers and len(offers) > 0:
                return offers[0]
            return None
        except json.JSONDecodeError:
            logger.error(f"Falha ao parsear resposta: {result.stdout}")
            return None
    
    async def _create_instance(
        self, 
        offer: dict, 
        backend: DedicatedBackend,
        model: str,
        config: DedicatedConfig
    ) -> int:
        """Cria instância no Vast.ai"""
        offer_id = offer["id"]
        
        # Gerar script onstart baseado no backend
        if backend == DedicatedBackend.OLLAMA:
            onstart_script = self._get_ollama_onstart_script(model)
            docker_image = "ollama/ollama:latest"
        else:
            onstart_script = self._get_vllm_onstart_script(model, config.quantization)
            docker_image = "vllm/vllm-openai:latest"
        
        result = await self._run_vastai_command([
            "create", "instance", str(offer_id),
            "--image", docker_image,
            "--disk", str(config.min_disk_gb),
            "--onstart-cmd", onstart_script,
            "--raw"
        ])
        
        if result.returncode != 0:
            raise RuntimeError(f"Falha ao criar instância: {result.stderr}")
        
        try:
            data = json.loads(result.stdout)
            instance_id = data.get("new_contract") or data.get("id")
            if not instance_id:
                raise RuntimeError(f"Resposta inválida: {result.stdout}")
            logger.info(f"Instância criada: {instance_id}")
            return int(instance_id)
        except (json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(f"Falha ao parsear resposta: {e}")
    
    def _get_ollama_onstart_script(self, model: str) -> str:
        """Gera script de inicialização para Ollama"""
        return f'''#!/bin/bash
# Start Ollama service
ollama serve &
sleep 5
# Pull the model
ollama pull {model}
# Keep container running
tail -f /dev/null
'''
    
    def _get_vllm_onstart_script(self, model: str, quantization: Optional[str]) -> str:
        """Gera script de inicialização para vLLM"""
        quant_arg = f"--quantization {quantization}" if quantization else ""
        return f'''#!/bin/bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model {model} \
    --host 0.0.0.0 \
    --port 8000 \
    {quant_arg}
'''
    
    async def _wait_for_instance(
        self, 
        instance_id: int, 
        timeout_seconds: int
    ) -> dict:
        """Aguarda instância ficar pronta e retorna info de conexão"""
        logger.info(f"Aguardando instância {instance_id} ficar pronta...")
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            result = await self._run_vastai_command([
                "show", "instance", str(instance_id), "--raw"
            ])
            
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    instance_data = data[0] if isinstance(data, list) else data
                    
                    status = instance_data.get("actual_status", "")
                    if status == "running":
                        # Extrair informações de conexão SSH
                        ssh_host = instance_data.get("public_ipaddr") or instance_data.get("ssh_host")
                        ssh_port = instance_data.get("ssh_port", 22)
                        
                        if ssh_host:
                            logger.info(f"Instância pronta: {ssh_host}:{ssh_port}")
                            return {"host": ssh_host, "port": ssh_port}
                    
                    logger.debug(f"Status atual: {status}")
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    logger.debug(f"Erro ao parsear status: {e}")
            
            await asyncio.sleep(10)
        
        raise RuntimeError(f"Timeout esperando instância {instance_id} ficar pronta")
    
    async def _wait_for_ollama_ready(
        self, 
        instance: DedicatedInstance, 
        model: str,
        timeout_minutes: int
    ):
        """Aguarda Ollama estar pronto e modelo carregado"""
        import httpx
        
        logger.info(f"Aguardando Ollama carregar modelo {model}...")
        
        timeout_seconds = timeout_minutes * 60
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(f"{instance.endpoint}/api/tags")
                    if response.status_code == 200:
                        data = response.json()
                        models = [m.get("name", "") for m in data.get("models", [])]
                        if any(model in m for m in models):
                            logger.info(f"Modelo {model} pronto!")
                            return
            except Exception as e:
                logger.debug(f"Ollama ainda não pronto: {e}")
            
            await asyncio.sleep(15)
        
        logger.warning(f"Timeout esperando modelo {model} carregar")
    
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
        
        # Verificar instâncias remotas no Vast.ai
        running_instances = await self._list_remote_instances()
        for inst_data in running_instances:
            if inst_data.get("actual_status") == "running":
                # Tentar usar instância existente
                instance_id = inst_data["id"]
                ssh_host = inst_data.get("public_ipaddr") or inst_data.get("ssh_host")
                ssh_port = inst_data.get("ssh_port", 22)
                
                if ssh_host:
                    instance = DedicatedInstance(
                        instance_id=instance_id,
                        backend=DedicatedBackend(backend),
                        model=model,
                        ssh_host=ssh_host,
                        ssh_port=ssh_port,
                        local_port=kwargs.get("local_port", 11435),
                        api_port=11434 if backend == "ollama" else 8000,
                    )
                    
                    if await instance.connect():
                        self._instances[instance_id] = instance
                        logger.info(f"Conectado a instância existente: {instance_id}")
                        return instance
        
        # Criar nova
        return await self.provision(model, backend, **kwargs)
    
    async def _list_remote_instances(self) -> list[dict]:
        """Lista instâncias remotas no Vast.ai"""
        result = await self._run_vastai_command(["show", "instances", "--raw"])
        
        if result.returncode != 0:
            return []
        
        try:
            return json.loads(result.stdout) or []
        except json.JSONDecodeError:
            return []
    
    def list_instances(self) -> list[DedicatedInstance]:
        """Lista instâncias ativas locais"""
        return list(self._instances.values())
    
    async def terminate(self, instance_id: int):
        """Termina uma instância"""
        # Desconectar túnel local
        if instance_id in self._instances:
            instance = self._instances[instance_id]
            instance.disconnect()
            del self._instances[instance_id]
        
        # Destruir no Vast.ai
        result = await self._run_vastai_command([
            "destroy", "instance", str(instance_id)
        ])
        
        if result.returncode == 0:
            logger.info(f"Instância {instance_id} terminada")
        else:
            logger.error(f"Falha ao terminar instância: {result.stderr}")
    
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
