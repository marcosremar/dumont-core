"""
Interface base para executores de teste.

Define o contrato que todos os executores devem seguir
para executar cenarios do plano unificado.
"""

from abc import ABC, abstractmethod
from typing import Optional

from dumont_core.testing.models.unified_plan import UnifiedTestPlan, TestScenario
from dumont_core.testing.models.test_result import ExecutionResult, ScenarioResult


class BaseExecutor(ABC):
    """
    Interface abstrata para executores de teste.

    Cada executor implementa sua propria estrategia de execucao,
    mas todos retornam resultados em formato padronizado.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        headless: bool = True,
        timeout: int = 60,
    ):
        """
        Inicializa o executor.

        Args:
            output_dir: Diretorio para salvar screenshots e logs
            headless: Se True, executa browser sem interface grafica
            timeout: Timeout padrao por cenario em segundos
        """
        self.output_dir = output_dir
        self.headless = headless
        self.timeout = timeout

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome identificador do executor."""
        pass

    @abstractmethod
    async def execute(self, plan: UnifiedTestPlan) -> ExecutionResult:
        """
        Executa todos os cenarios do plano.

        Args:
            plan: Plano de testes unificado

        Returns:
            ExecutionResult com resultados de todos os cenarios
        """
        pass

    @abstractmethod
    async def execute_scenario(self, scenario: TestScenario, base_url: str) -> ScenarioResult:
        """
        Executa um cenario individual.

        Args:
            scenario: Cenario a executar
            base_url: URL base da aplicacao

        Returns:
            ScenarioResult com resultado do cenario
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
