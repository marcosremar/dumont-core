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

from .base_executor import BaseExecutor
from .browseruse_executor import BrowserUseExecutor
from .playwright_executor import PlaywrightExecutor
from .results_consolidator import ResultsConsolidator

__all__ = [
    "BaseExecutor",
    "BrowserUseExecutor",
    "PlaywrightExecutor",
    "ResultsConsolidator",
]
