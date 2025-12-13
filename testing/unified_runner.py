"""
Orquestrador principal do sistema de testes colaborativos.

Coordena as fases de exploracao e execucao, usando tanto
Browser-Use quanto Playwright para descoberta e testes.
"""

from typing import Optional
from pathlib import Path
from datetime import datetime

from dumont_core.testing.models.unified_plan import UnifiedTestPlan
from dumont_core.testing.models.test_result import ConsolidatedReport

from dumont_core.testing.exploration.browseruse_explorer import BrowserUseExplorer
from dumont_core.testing.exploration.playwright_explorer import PlaywrightExplorer
from dumont_core.testing.exploration.discovery_merger import DiscoveryMerger

from dumont_core.testing.execution.browseruse_executor import BrowserUseExecutor
from dumont_core.testing.execution.playwright_executor import PlaywrightExecutor
from dumont_core.testing.execution.results_consolidator import ResultsConsolidator


class UnifiedTestRunner:
    """
    Orquestrador principal para testes colaborativos.

    Coordena:
    - Fase 1: Exploracao colaborativa (Browser-Use + Playwright)
    - Fase 2: Execucao de testes (Browser-Use + Playwright)
    - Consolidacao de resultados
    """

    def __init__(
        self,
        base_url: str,
        app_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        llm_model: str = "google/gemini-2.5-flash",
        llm_provider: str = "openrouter",
        headless: bool = True,
    ):
        """
        Inicializa o runner.

        Args:
            base_url: URL base da aplicacao a testar
            app_name: Nome da aplicacao (para relatorios)
            output_dir: Diretorio para salvar resultados
            llm_model: Modelo LLM para Browser-Use
            llm_provider: Provider do LLM (openrouter, anthropic, openai)
            headless: Se True, executa browsers sem interface
        """
        self.base_url = base_url
        self.app_name = app_name or self._extract_app_name(base_url)
        self.output_dir = output_dir
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.headless = headless

        # Cria diretorio de saida se especificado
        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _extract_app_name(self, url: str) -> str:
        """Extrai nome da app a partir da URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc or "app"

    async def explore(
        self,
        explorers: list[str] = None,
        timeout: int = 300,
    ) -> UnifiedTestPlan:
        """
        Fase 1: Exploracao colaborativa.

        Executa exploradores selecionados e consolida descobertas
        em um plano de testes unificado.

        Args:
            explorers: Lista de exploradores a usar ["browseruse", "playwright"]
                      Se None, usa ambos
            timeout: Timeout para cada explorador em segundos

        Returns:
            UnifiedTestPlan com features e cenarios descobertos
        """
        if explorers is None:
            explorers = ["browseruse", "playwright"]

        browseruse_result = None
        playwright_result = None

        # Executa Browser-Use Explorer
        if "browseruse" in explorers:
            print(f"[Explore] Iniciando Browser-Use Explorer...")
            explorer = BrowserUseExplorer(
                base_url=self.base_url,
                output_dir=self.output_dir,
                headless=self.headless,
                timeout=timeout,
                llm_model=self.llm_model,
                llm_provider=self.llm_provider,
            )
            try:
                browseruse_result = await explorer.explore()
                print(f"[Explore] Browser-Use: {len(browseruse_result.features)} features, {len(browseruse_result.user_flows)} fluxos")
            except Exception as e:
                print(f"[Explore] Erro no Browser-Use: {e}")

        # Executa Playwright Explorer
        if "playwright" in explorers:
            print(f"[Explore] Iniciando Playwright Explorer...")
            explorer = PlaywrightExplorer(
                base_url=self.base_url,
                output_dir=self.output_dir,
                headless=self.headless,
                timeout=timeout,
            )
            try:
                playwright_result = await explorer.explore()
                print(f"[Explore] Playwright: {len(playwright_result.features)} features, {len(playwright_result.api_endpoints)} APIs")
            except Exception as e:
                print(f"[Explore] Erro no Playwright: {e}")

        # Consolida descobertas
        print(f"[Explore] Consolidando descobertas...")
        merger = DiscoveryMerger(
            app_name=self.app_name,
            base_url=self.base_url,
        )
        plan = merger.merge(browseruse_result, playwright_result)

        print(f"[Explore] Plano gerado: {len(plan.features)} features, {len(plan.test_scenarios)} cenarios")

        # Salva plano se output_dir especificado
        if self.output_dir:
            plan_path = Path(self.output_dir) / "test_plan.json"
            plan.save(str(plan_path))
            print(f"[Explore] Plano salvo em: {plan_path}")

        return plan

    async def execute(
        self,
        plan: UnifiedTestPlan,
        executors: list[str] = None,
    ) -> ConsolidatedReport:
        """
        Fase 2: Execucao de testes.

        Executa cenarios do plano usando executores selecionados
        e consolida resultados.

        Args:
            plan: Plano de testes a executar
            executors: Lista de executores a usar ["browseruse", "playwright"]
                      Se None, usa ambos

        Returns:
            ConsolidatedReport com resultados comparados
        """
        if executors is None:
            executors = ["browseruse", "playwright"]

        browseruse_result = None
        playwright_result = None

        # Executa com Browser-Use
        if "browseruse" in executors:
            print(f"[Execute] Iniciando Browser-Use Executor...")
            executor = BrowserUseExecutor(
                output_dir=self.output_dir,
                headless=self.headless,
                llm_model=self.llm_model,
                llm_provider=self.llm_provider,
            )
            try:
                browseruse_result = await executor.execute(plan)
                print(f"[Execute] Browser-Use: {browseruse_result.passed_scenarios}/{browseruse_result.total_scenarios} passaram")
            except Exception as e:
                print(f"[Execute] Erro no Browser-Use: {e}")

        # Executa com Playwright
        if "playwright" in executors:
            print(f"[Execute] Iniciando Playwright Executor...")
            executor = PlaywrightExecutor(
                output_dir=self.output_dir,
                headless=self.headless,
            )
            try:
                playwright_result = await executor.execute(plan)
                print(f"[Execute] Playwright: {playwright_result.passed_scenarios}/{playwright_result.total_scenarios} passaram")
            except Exception as e:
                print(f"[Execute] Erro no Playwright: {e}")

        # Consolida resultados
        print(f"[Execute] Consolidando resultados...")
        consolidator = ResultsConsolidator(
            app_name=self.app_name,
            base_url=self.base_url,
        )
        report = consolidator.consolidate(browseruse_result, playwright_result)

        print(f"[Execute] Taxa de sucesso: {report.overall_pass_rate:.1f}%")

        # Salva relatorio se output_dir especificado
        if self.output_dir:
            report_path = Path(self.output_dir) / "test_report.json"
            report.save(str(report_path))

            html_path = Path(self.output_dir) / "test_report.html"
            report.export_html(str(html_path))

            print(f"[Execute] Relatorios salvos em: {self.output_dir}")

        return report

    async def run_full(
        self,
        explorers: list[str] = None,
        executors: list[str] = None,
        exploration_timeout: int = 300,
    ) -> ConsolidatedReport:
        """
        Executa ciclo completo: exploracao + testes.

        Args:
            explorers: Exploradores a usar na fase 1
            executors: Executores a usar na fase 2
            exploration_timeout: Timeout para exploracao

        Returns:
            ConsolidatedReport com resultados finais
        """
        print(f"\n{'='*60}")
        print(f"TESTE COMPLETO - {self.app_name}")
        print(f"{'='*60}")
        print(f"URL: {self.base_url}")
        print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        # Fase 1: Exploracao
        print("[FASE 1] EXPLORACAO COLABORATIVA")
        print("-" * 40)
        plan = await self.explore(explorers=explorers, timeout=exploration_timeout)

        print()

        # Fase 2: Execucao
        print("[FASE 2] EXECUCAO DE TESTES")
        print("-" * 40)
        report = await self.execute(plan=plan, executors=executors)

        print()
        print(f"{'='*60}")
        print("RESULTADO FINAL")
        print(f"{'='*60}")
        print(f"Features descobertas: {len(plan.features)}")
        print(f"Cenarios testados: {report.total_scenarios_tested}")
        print(f"Taxa de sucesso: {report.overall_pass_rate:.1f}%")
        print(f"Discrepancias: {report.scenarios_with_discrepancy}")
        print(f"{'='*60}\n")

        return report

    async def run_exploration_only(
        self,
        explorers: list[str] = None,
        timeout: int = 300,
    ) -> UnifiedTestPlan:
        """
        Executa apenas a fase de exploracao.

        Util para gerar plano de testes que sera executado depois,
        ou para analise manual das descobertas.

        Args:
            explorers: Exploradores a usar
            timeout: Timeout para exploracao

        Returns:
            UnifiedTestPlan com descobertas
        """
        return await self.explore(explorers=explorers, timeout=timeout)

    async def run_execution_only(
        self,
        plan_path: str,
        executors: list[str] = None,
    ) -> ConsolidatedReport:
        """
        Executa apenas a fase de testes usando plano existente.

        Util para re-executar testes sem re-explorar,
        ideal para CI/CD.

        Args:
            plan_path: Caminho para arquivo JSON do plano
            executors: Executores a usar

        Returns:
            ConsolidatedReport com resultados
        """
        plan = UnifiedTestPlan.load(plan_path)
        return await self.execute(plan=plan, executors=executors)
