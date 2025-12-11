#!/usr/bin/env python3
"""
LLM Manager - Gerenciador Unificado de Conexões LLM
====================================================
Gerencia conexões com diferentes provedores de LLM:
- OpenRouter (via API)
- Ollama Local
- Ollama Remoto (via túnel SSH automático para VPS com GPU)

Features:
- Gerenciamento automático de túnel SSH para servidores remotos
- Fallback entre provedores
- Health checks automáticos
- Cache de conexões
- Suporte a múltiplos modelos
"""

import os
import asyncio
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import httpx
from dotenv import load_dotenv

from .tunnel import SSHTunnel, RemoteServerConfig

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logger = logging.getLogger("dumont.llm.manager")


class LLMProvider(Enum):
    """Provedores de LLM suportados"""
    OPENROUTER = "openrouter"
    LITELLM = "litellm"  # Alias para OpenRouter via LiteLLM
    OLLAMA_LOCAL = "ollama"
    OLLAMA_REMOTE = "ollama-remote"


@dataclass
class LLMConfig:
    """Configuração de um LLM"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 120
    extra_params: Dict[str, Any] = field(default_factory=dict)


class LLMManager:
    """
    Gerenciador unificado de conexões LLM.

    Suporta múltiplos provedores:
    - OpenRouter (API cloud)
    - Ollama local
    - Ollama remoto (via túnel SSH automático para VPS com GPU)
    """

    def __init__(self):
        self._tunnels: Dict[str, SSHTunnel] = {}
        self._health_cache: Dict[str, tuple] = {}  # (status, timestamp)
        self._llm_cache: Dict[str, Any] = {}

        # Configurações padrão
        self.default_models = {
            LLMProvider.OPENROUTER: os.environ.get("OPENROUTER_DEFAULT_MODEL", "openai/gpt-4o"),
            LLMProvider.OLLAMA_LOCAL: os.environ.get("OLLAMA_DEFAULT_MODEL", "qwen2.5-coder:14b"),
            LLMProvider.OLLAMA_REMOTE: os.environ.get("OLLAMA_DEFAULT_MODEL", "qwen2.5-coder:14b"),
        }

        # Carregar configuração de servidor remoto
        self.remote_config = RemoteServerConfig.from_env()

        logger.info("LLMManager inicializado")
        self._log_available_providers()

    def _log_available_providers(self):
        """Loga os provedores disponíveis"""
        providers = []

        if os.environ.get("OPENROUTER_API_KEY"):
            providers.append("OpenRouter")

        if self._check_ollama_local():
            providers.append("Ollama Local")

        if self.remote_config:
            providers.append(f"Ollama Remoto ({self.remote_config.host})")

        if providers:
            logger.info(f"Provedores disponíveis: {', '.join(providers)}")
        else:
            logger.warning("Nenhum provedor LLM configurado!")

    def _check_ollama_local(self) -> bool:
        """Verifica se há Ollama local rodando"""
        try:
            response = httpx.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    async def get_llm(
        self,
        provider: str = "auto",
        model: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Obtém uma instância de LLM configurada.

        Args:
            provider: Provedor a usar ("openrouter", "ollama", "ollama-remote", "auto")
            model: Modelo a usar (usa padrão se não especificado)
            **kwargs: Parâmetros extras para o LLM

        Returns:
            Instância do LLM (ChatOpenAI ou ChatOllama)
        """
        # Resolver provider "auto"
        if provider == "auto":
            provider = await self._auto_select_provider()

        provider_enum = LLMProvider(provider)

        # Usar modelo padrão se não especificado
        if model is None:
            model = self.default_models.get(provider_enum, "gpt-4o")

        cache_key = f"{provider}:{model}"

        # Verificar cache (com health check)
        if cache_key in self._llm_cache:
            if await self._verify_llm_health(provider_enum):
                return self._llm_cache[cache_key]

        # Criar novo LLM
        llm = await self._create_llm(provider_enum, model, **kwargs)
        self._llm_cache[cache_key] = llm

        return llm

    async def _auto_select_provider(self) -> str:
        """Seleciona automaticamente o melhor provedor disponível"""
        # Prioridade: Ollama Local > Ollama Remoto > OpenRouter

        # 1. Tentar Ollama local
        if self._check_ollama_local():
            logger.info("Auto-selecionado: Ollama Local")
            return "ollama"

        # 2. Tentar Ollama remoto
        if self.remote_config:
            if await self._ensure_remote_tunnel():
                logger.info("Auto-selecionado: Ollama Remoto")
                return "ollama-remote"

        # 3. Fallback para OpenRouter
        if os.environ.get("OPENROUTER_API_KEY"):
            logger.info("Auto-selecionado: OpenRouter")
            return "openrouter"

        raise RuntimeError("Nenhum provedor LLM disponível!")

    async def _create_llm(
        self,
        provider: LLMProvider,
        model: str,
        **kwargs
    ) -> Any:
        """Cria instância do LLM baseado no provedor"""

        if provider == LLMProvider.OPENROUTER or provider == LLMProvider.LITELLM:
            return await self._create_openrouter_llm(model, **kwargs)

        elif provider == LLMProvider.OLLAMA_LOCAL:
            return await self._create_ollama_llm(model, "http://localhost:11434", **kwargs)

        elif provider == LLMProvider.OLLAMA_REMOTE:
            # Garantir que o túnel está ativo
            if not await self._ensure_remote_tunnel():
                raise RuntimeError("Não foi possível estabelecer túnel SSH para servidor remoto")

            local_port = self.remote_config.local_port if self.remote_config else 11434
            return await self._create_ollama_llm(model, f"http://localhost:{local_port}", **kwargs)

        else:
            raise ValueError(f"Provedor não suportado: {provider}")

    async def _create_openrouter_llm(self, model: str, **kwargs) -> Any:
        """Cria LLM via OpenRouter"""
        # Importar dinamicamente para evitar dependência obrigatória
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            try:
                from browser_use import ChatOpenAI
            except ImportError:
                raise RuntimeError("langchain_openai ou browser_use não instalado")

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY não configurada")

        logger.info(f"Criando LLM OpenRouter: {model}")

        return ChatOpenAI(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            **kwargs
        )

    async def _create_ollama_llm(self, model: str, host: str, **kwargs) -> Any:
        """Cria LLM via Ollama"""
        # Importar dinamicamente para evitar dependência obrigatória
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            try:
                from browser_use import ChatOllama
            except ImportError:
                raise RuntimeError("langchain_ollama ou browser_use não instalado")

        logger.info(f"Criando LLM Ollama: {model} @ {host}")

        return ChatOllama(
            model=model,
            base_url=host,
            timeout=kwargs.pop("timeout", 120),
            **kwargs
        )

    async def _ensure_remote_tunnel(self) -> bool:
        """Garante que o túnel SSH está ativo"""
        if not self.remote_config:
            return False

        tunnel_key = f"{self.remote_config.host}:{self.remote_config.ssh_port}"

        # Verificar túnel existente
        if tunnel_key in self._tunnels:
            tunnel = self._tunnels[tunnel_key]
            if tunnel.is_active():
                return True

        # Criar novo túnel
        tunnel = SSHTunnel(self.remote_config)
        if tunnel.start():
            self._tunnels[tunnel_key] = tunnel
            return True

        return False

    async def _verify_llm_health(self, provider: LLMProvider) -> bool:
        """Verifica saúde do LLM"""
        cache_key = provider.value
        now = datetime.now().timestamp()

        # Verificar cache (válido por 30 segundos)
        if cache_key in self._health_cache:
            status, timestamp = self._health_cache[cache_key]
            if now - timestamp < 30:
                return status

        # Fazer health check
        try:
            if provider == LLMProvider.OPENROUTER:
                status = bool(os.environ.get("OPENROUTER_API_KEY"))

            elif provider == LLMProvider.OLLAMA_LOCAL:
                status = self._check_ollama_local()

            elif provider == LLMProvider.OLLAMA_REMOTE:
                if self.remote_config:
                    local_port = self.remote_config.local_port
                    response = httpx.get(f"http://localhost:{local_port}/api/tags", timeout=5)
                    status = response.status_code == 200
                else:
                    status = False

            else:
                status = False

            self._health_cache[cache_key] = (status, now)
            return status

        except Exception as e:
            logger.debug(f"Health check falhou para {provider}: {e}")
            self._health_cache[cache_key] = (False, now)
            return False

    async def list_models(self, provider: str = "ollama") -> List[Dict[str, Any]]:
        """Lista modelos disponíveis em um provedor"""
        provider_enum = LLMProvider(provider)

        if provider_enum in (LLMProvider.OLLAMA_LOCAL, LLMProvider.OLLAMA_REMOTE):
            return await self._list_ollama_models(provider_enum)

        elif provider_enum == LLMProvider.OPENROUTER:
            # OpenRouter não tem endpoint de listagem fácil
            # Retornar modelos comuns
            return [
                {"name": "openai/gpt-4o", "size": "cloud"},
                {"name": "openai/gpt-4o-mini", "size": "cloud"},
                {"name": "anthropic/claude-3.5-sonnet", "size": "cloud"},
                {"name": "google/gemini-2.0-flash-exp", "size": "cloud"},
            ]

        return []

    async def _list_ollama_models(self, provider: LLMProvider) -> List[Dict[str, Any]]:
        """Lista modelos Ollama"""
        if provider == LLMProvider.OLLAMA_REMOTE:
            if not await self._ensure_remote_tunnel():
                return []
            local_port = self.remote_config.local_port if self.remote_config else 11434
            base_url = f"http://localhost:{local_port}"
        else:
            base_url = "http://localhost:11434"

        try:
            response = httpx.get(f"{base_url}/api/tags", timeout=10)
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            logger.error(f"Erro ao listar modelos Ollama: {e}")
            return []

    async def pull_model(self, model: str, provider: str = "ollama") -> bool:
        """Baixa um modelo para o Ollama"""
        provider_enum = LLMProvider(provider)

        if provider_enum not in (LLMProvider.OLLAMA_LOCAL, LLMProvider.OLLAMA_REMOTE):
            logger.error("pull_model só funciona com Ollama")
            return False

        if provider_enum == LLMProvider.OLLAMA_REMOTE:
            if not await self._ensure_remote_tunnel():
                return False
            local_port = self.remote_config.local_port if self.remote_config else 11434
            base_url = f"http://localhost:{local_port}"
        else:
            base_url = "http://localhost:11434"

        try:
            logger.info(f"Baixando modelo {model}...")

            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{base_url}/api/pull",
                    json={"name": model, "stream": True}
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                status = data.get("status", "")
                                if "pulling" in status:
                                    completed = data.get("completed", 0)
                                    total = data.get("total", 0)
                                    if total > 0:
                                        pct = (completed / total) * 100
                                        print(f"\r{status}: {pct:.1f}%", end="", flush=True)
                            except json.JSONDecodeError:
                                pass

            print()  # Nova linha
            logger.info(f"Modelo {model} baixado com sucesso!")
            return True

        except Exception as e:
            logger.error(f"Erro ao baixar modelo: {e}")
            return False

    def cleanup(self):
        """Limpa recursos (fecha túneis SSH)"""
        for tunnel in self._tunnels.values():
            tunnel.stop()
        self._tunnels.clear()
        self._llm_cache.clear()
        logger.info("LLMManager cleanup completo")

    def __del__(self):
        self.cleanup()


# Singleton global
_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Obtém instância singleton do LLMManager"""
    global _manager
    if _manager is None:
        _manager = LLMManager()
    return _manager


# Funções de conveniência
async def get_llm(provider: str = "auto", model: Optional[str] = None, **kwargs) -> Any:
    """Atalho para obter um LLM"""
    return await get_llm_manager().get_llm(provider, model, **kwargs)


async def list_models(provider: str = "ollama") -> List[Dict[str, Any]]:
    """Atalho para listar modelos"""
    return await get_llm_manager().list_models(provider)


# CLI para testes
async def main():
    """CLI para testar o LLMManager"""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Manager CLI")
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")

    # Comando: list
    list_parser = subparsers.add_parser("list", help="Listar modelos")
    list_parser.add_argument("--provider", "-p", default="ollama", help="Provedor (ollama, openrouter)")

    # Comando: test
    test_parser = subparsers.add_parser("test", help="Testar conexão")
    test_parser.add_argument("--provider", "-p", default="auto", help="Provedor")
    test_parser.add_argument("--model", "-m", help="Modelo")
    test_parser.add_argument("--prompt", default="Olá! Responda com 'Funcionando!'", help="Prompt de teste")

    # Comando: pull
    pull_parser = subparsers.add_parser("pull", help="Baixar modelo")
    pull_parser.add_argument("model", help="Nome do modelo")
    pull_parser.add_argument("--provider", "-p", default="ollama", help="Provedor")

    args = parser.parse_args()

    manager = get_llm_manager()

    try:
        if args.command == "list":
            print(f"\nModelos disponíveis ({args.provider}):")
            print("=" * 50)
            models = await manager.list_models(args.provider)
            for m in models:
                name = m.get("name", m)
                size = m.get("size", "?")
                print(f"  - {name} ({size})")
            print()

        elif args.command == "test":
            print(f"\nTestando conexão ({args.provider})...")
            print("=" * 50)

            llm = await manager.get_llm(args.provider, args.model)
            print(f"LLM criado: {type(llm).__name__}")

            # Fazer chamada de teste
            response = await llm.ainvoke(args.prompt)
            content = getattr(response, 'content', str(response))
            print(f"\nResposta: {content}")
            print("\n✅ Teste bem-sucedido!")

        elif args.command == "pull":
            await manager.pull_model(args.model, args.provider)

        else:
            parser.print_help()

    finally:
        manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
