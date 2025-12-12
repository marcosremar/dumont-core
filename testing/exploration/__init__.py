"""
Modulo de exploracao colaborativa.

Contem os exploradores que descobrem features, fluxos e edge cases
antes da fase de execucao de testes.

Exporta:
- BaseExplorer: Interface abstrata para exploradores
- BrowserUseExplorer: Exploracao com Browser-Use + LLM
- PlaywrightExplorer: Exploracao com Playwright Agents
- DiscoveryMerger: Consolida descobertas em plano unificado
"""

from .base_explorer import BaseExplorer
from .browseruse_explorer import BrowserUseExplorer
from .playwright_explorer import PlaywrightExplorer
from .discovery_merger import DiscoveryMerger

__all__ = [
    "BaseExplorer",
    "BrowserUseExplorer",
    "PlaywrightExplorer",
    "DiscoveryMerger",
]
