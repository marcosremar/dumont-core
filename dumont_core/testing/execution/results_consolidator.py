"""
Consolidador de resultados de multiplos executores.

Combina resultados do Browser-Use e Playwright,
identificando discrepancias e gerando relatorio final.
"""

from datetime import datetime
from typing import Optional

from dumont_core.testing.models.test_result import (
    ExecutionResult,
    ConsolidatedReport,
    Discrepancy,
    ScenarioStatus,
)


class ResultsConsolidator:
    """
    Combina e analisa resultados de multiplos executores.

    Funcoes:
    - Compara resultados do mesmo cenario entre executores
    - Identifica discrepancias (passou em um, falhou em outro)
    - Calcula metricas consolidadas
    - Gera relatorio final
    """

    def __init__(self, app_name: str, base_url: str):
        self.app_name = app_name
        self.base_url = base_url

    def consolidate(
        self,
        browseruse_result: Optional[ExecutionResult] = None,
        playwright_result: Optional[ExecutionResult] = None,
    ) -> ConsolidatedReport:
        """
        Consolida resultados de ambos executores.

        Args:
            browseruse_result: Resultado do Browser-Use Executor
            playwright_result: Resultado do Playwright Executor

        Returns:
            ConsolidatedReport com analise comparativa
        """
        discrepancies = []
        ux_insights = []

        # Coleta IDs de cenarios de ambos
        bu_scenarios = {}
        pw_scenarios = {}

        if browseruse_result:
            for r in browseruse_result.scenario_results:
                bu_scenarios[r.scenario_id] = r
                # Coleta UX insights
                if r.ux_summary:
                    ux_insights.append(f"{r.scenario_name}: {r.ux_summary}")

        if playwright_result:
            for r in playwright_result.scenario_results:
                pw_scenarios[r.scenario_id] = r

        # Encontra todos os cenarios testados
        all_scenario_ids = set(bu_scenarios.keys()) | set(pw_scenarios.keys())

        # Analisa cada cenario
        passed_both = 0
        failed_both = 0
        with_discrepancy = 0

        for scenario_id in all_scenario_ids:
            bu_result = bu_scenarios.get(scenario_id)
            pw_result = pw_scenarios.get(scenario_id)

            bu_status = bu_result.status if bu_result else None
            pw_status = pw_result.status if pw_result else None

            # Ambos passaram
            if bu_status == ScenarioStatus.PASSED and pw_status == ScenarioStatus.PASSED:
                passed_both += 1

            # Ambos falharam
            elif bu_status in [ScenarioStatus.FAILED, ScenarioStatus.ERROR] and \
                 pw_status in [ScenarioStatus.FAILED, ScenarioStatus.ERROR]:
                failed_both += 1

            # Discrepancia
            elif bu_status and pw_status and bu_status != pw_status:
                with_discrepancy += 1
                scenario_name = bu_result.scenario_name if bu_result else pw_result.scenario_name

                # Descreve a discrepancia
                if bu_status == ScenarioStatus.PASSED:
                    description = "Passou no Browser-Use mas falhou no Playwright. Possivel problema de seletor."
                else:
                    description = "Passou no Playwright mas falhou no Browser-Use. Possivel problema de UX ou timing."

                discrepancies.append(Discrepancy(
                    scenario_id=scenario_id,
                    scenario_name=scenario_name,
                    browseruse_status=bu_status.value if bu_status else None,
                    playwright_status=pw_status.value if pw_status else None,
                    description=description,
                ))

            # Apenas um executor testou
            else:
                if bu_status == ScenarioStatus.PASSED or pw_status == ScenarioStatus.PASSED:
                    passed_both += 1  # Conta como passou se pelo menos um passou
                else:
                    failed_both += 1

        return ConsolidatedReport(
            timestamp=datetime.now(),
            base_url=self.base_url,
            app_name=self.app_name,
            browseruse_result=browseruse_result,
            playwright_result=playwright_result,
            discrepancies=discrepancies,
            total_scenarios_tested=len(all_scenario_ids),
            scenarios_passed_both=passed_both,
            scenarios_failed_both=failed_both,
            scenarios_with_discrepancy=with_discrepancy,
            ux_insights=ux_insights,
        )

    def analyze_discrepancy(self, discrepancy: Discrepancy) -> dict:
        """
        Analisa uma discrepancia em mais detalhes.

        Args:
            discrepancy: Discrepancia a analisar

        Returns:
            dict com analise detalhada
        """
        analysis = {
            "scenario": discrepancy.scenario_name,
            "likely_cause": "",
            "recommendation": "",
        }

        bu = discrepancy.browseruse_status
        pw = discrepancy.playwright_status

        if bu == "passed" and pw in ["failed", "error"]:
            analysis["likely_cause"] = "Seletor do Playwright pode estar incorreto ou desatualizado"
            analysis["recommendation"] = (
                "1. Verifique se o seletor ainda e valido\n"
                "2. Use seletores mais robustos (data-testid, aria-label)\n"
                "3. Considere aumentar o timeout"
            )

        elif pw == "passed" and bu in ["failed", "error"]:
            analysis["likely_cause"] = "Browser-Use pode ter encontrado problema de UX ou timing"
            analysis["recommendation"] = (
                "1. Revise o feedback de UX do Browser-Use\n"
                "2. Verifique se ha delays necessarios\n"
                "3. Considere simplificar o fluxo de teste"
            )

        return analysis

    def generate_summary(self, report: ConsolidatedReport) -> str:
        """
        Gera resumo textual do relatorio.

        Args:
            report: Relatorio consolidado

        Returns:
            String com resumo
        """
        lines = [
            f"# Relatorio de Testes - {report.app_name}",
            f"",
            f"**Data:** {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**URL:** {report.base_url}",
            f"",
            "## Resultados",
            f"",
            f"- Total de cenarios: {report.total_scenarios_tested}",
            f"- Passaram (ambos): {report.scenarios_passed_both}",
            f"- Falharam (ambos): {report.scenarios_failed_both}",
            f"- Com discrepancia: {report.scenarios_with_discrepancy}",
            f"- Taxa de sucesso: {report.overall_pass_rate:.1f}%",
            f"",
        ]

        if report.browseruse_result:
            lines.extend([
                "### Browser-Use",
                f"- Total: {report.browseruse_result.total_scenarios}",
                f"- Passaram: {report.browseruse_result.passed_scenarios}",
                f"- Falharam: {report.browseruse_result.failed_scenarios}",
                f"- Taxa: {report.browseruse_result.pass_rate:.1f}%",
                f"",
            ])

        if report.playwright_result:
            lines.extend([
                "### Playwright",
                f"- Total: {report.playwright_result.total_scenarios}",
                f"- Passaram: {report.playwright_result.passed_scenarios}",
                f"- Falharam: {report.playwright_result.failed_scenarios}",
                f"- Taxa: {report.playwright_result.pass_rate:.1f}%",
                f"",
            ])

        if report.discrepancies:
            lines.extend([
                "## Discrepancias",
                f"",
            ])
            for d in report.discrepancies:
                lines.extend([
                    f"### {d.scenario_name}",
                    f"- Browser-Use: {d.browseruse_status}",
                    f"- Playwright: {d.playwright_status}",
                    f"- Analise: {d.description}",
                    f"",
                ])

        if report.ux_insights:
            lines.extend([
                "## Insights de UX",
                f"",
            ])
            for insight in report.ux_insights:
                lines.append(f"- {insight}")

        return "\n".join(lines)
