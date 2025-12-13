"""
Modulo de execucao de testes.

Contem os executores que rodam os cenarios de teste do plano unificado
e o consolidador de resultados.

Exporta:
- BaseExecutor: Interface abstrata para executores
- BrowserUseExecutor: Execucao com Browser-Use + LLM
- PlaywrightExecutor: Execucao com Playwright
- ResultsConsolidator: Consolida resultados de multiplos executores
"""

from dumont_core.testing.execution.base_executor import BaseExecutor
from dumont_core.testing.execution.browseruse_executor import BrowserUseExecutor
from dumont_core.testing.execution.playwright_executor import PlaywrightExecutor
from dumont_core.testing.execution.results_consolidator import ResultsConsolidator

__all__ = [
    "BaseExecutor",
    "BrowserUseExecutor",
    "PlaywrightExecutor",
    "ResultsConsolidator",
]
