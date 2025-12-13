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

from dumont_core.testing.exploration.base_explorer import BaseExplorer
from dumont_core.testing.exploration.browseruse_explorer import BrowserUseExplorer
from dumont_core.testing.exploration.playwright_explorer import PlaywrightExplorer
from dumont_core.testing.exploration.discovery_merger import DiscoveryMerger

__all__ = [
    "BaseExplorer",
    "BrowserUseExplorer",
    "PlaywrightExplorer",
    "DiscoveryMerger",
]
