"""
Interface base para exploradores.

Define o contrato que todos os exploradores devem seguir
para participar da exploracao colaborativa.
"""

from abc import ABC, abstractmethod
from typing import Optional

from dumont_core.testing.models.discovery import DiscoveryResult


class BaseExplorer(ABC):
    """
    Interface abstrata para exploradores de aplicacao.

    Cada explorador implementa sua propria estrategia de descoberta,
    mas todos retornam um DiscoveryResult padronizado.
    """

    def __init__(
        self,
        base_url: str,
        output_dir: Optional[str] = None,
        headless: bool = True,
        timeout: int = 300,
    ):
        """
        Inicializa o explorador.

        Args:
            base_url: URL base da aplicacao a explorar
            output_dir: Diretorio para salvar screenshots e logs
            headless: Se True, executa browser sem interface grafica
            timeout: Timeout maximo para exploracao em segundos
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.headless = headless
        self.timeout = timeout

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome identificador do explorador."""
        pass

    @abstractmethod
    async def explore(self) -> DiscoveryResult:
        """
        Executa exploracao da aplicacao.

        Returns:
            DiscoveryResult com todas as descobertas
        """
        pass

    async def __aenter__(self):
        """Suporte a context manager async."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup ao sair do context manager."""
        await self.cleanup()

    async def cleanup(self):
        """
        Limpa recursos alocados.

        Implementacoes devem sobrescrever para fechar browsers, etc.
        """
        pass
