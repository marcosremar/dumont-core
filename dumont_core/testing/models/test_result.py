"""
Modelos de dados para resultados de execucao de testes.

Contem as estruturas usadas pelos executores para registrar
resultados e pelo consolidador para gerar o relatorio final.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
import json


class StepStatus(str, Enum):
    """Status de execucao de um step."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class ScenarioStatus(str, Enum):
    """Status de execucao de um cenario."""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """
    Resultado da execucao de um step individual.
    """
    step_index: int
    action: str
    status: StepStatus
    duration_ms: int = 0

    # Detalhes
    message: Optional[str] = None
    error: Optional[str] = None
    screenshot: Optional[str] = None

    # UX feedback (do Browser-Use)
    ux_feedback: Optional[str] = None

    def to_dict(self) -> dict:
        """Converte para dicionario."""
        return {
            "step_index": self.step_index,
            "action": self.action,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "message": self.message,
            "error": self.error,
            "screenshot": self.screenshot,
            "ux_feedback": self.ux_feedback,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StepResult":
        """Cria a partir de dicionario."""
        return cls(
            step_index=data["step_index"],
            action=data["action"],
            status=StepStatus(data["status"]),
            duration_ms=data.get("duration_ms", 0),
            message=data.get("message"),
            error=data.get("error"),
            screenshot=data.get("screenshot"),
            ux_feedback=data.get("ux_feedback"),
        )


@dataclass
class ScenarioResult:
    """
    Resultado da execucao de um cenario de teste.
    """
    scenario_id: str
    scenario_name: str
    status: ScenarioStatus
    executor: str  # "browseruse" ou "playwright"

    # Resultados dos steps
    step_results: list[StepResult] = field(default_factory=list)

    # Metricas
    duration_ms: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Detalhes
    error: Optional[str] = None
    screenshots: list[str] = field(default_factory=list)

    # UX feedback consolidado (do Browser-Use)
    ux_summary: Optional[str] = None

    @property
    def passed_steps(self) -> int:
        """Numero de steps que passaram."""
        return sum(1 for s in self.step_results if s.status == StepStatus.PASSED)

    @property
    def failed_steps(self) -> int:
        """Numero de steps que falharam."""
        return sum(1 for s in self.step_results if s.status == StepStatus.FAILED)

    def to_dict(self) -> dict:
        """Converte para dicionario."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "status": self.status.value,
            "executor": self.executor,
            "step_results": [s.to_dict() for s in self.step_results],
            "duration_ms": self.duration_ms,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": self.error,
            "screenshots": self.screenshots,
            "ux_summary": self.ux_summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioResult":
        """Cria a partir de dicionario."""
        return cls(
            scenario_id=data["scenario_id"],
            scenario_name=data["scenario_name"],
            status=ScenarioStatus(data["status"]),
            executor=data["executor"],
            step_results=[StepResult.from_dict(s) for s in data.get("step_results", [])],
            duration_ms=data.get("duration_ms", 0),
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            error=data.get("error"),
            screenshots=data.get("screenshots", []),
            ux_summary=data.get("ux_summary"),
        )


@dataclass
class ExecutionResult:
    """
    Resultado da execucao de todos os cenarios por um executor.
    """
    executor: str  # "browseruse" ou "playwright"
    timestamp: datetime
    base_url: str

    # Resultados
    scenario_results: list[ScenarioResult] = field(default_factory=list)

    # Metricas agregadas
    total_duration_ms: int = 0

    @property
    def total_scenarios(self) -> int:
        """Total de cenarios executados."""
        return len(self.scenario_results)

    @property
    def passed_scenarios(self) -> int:
        """Cenarios que passaram."""
        return sum(1 for s in self.scenario_results if s.status == ScenarioStatus.PASSED)

    @property
    def failed_scenarios(self) -> int:
        """Cenarios que falharam."""
        return sum(1 for s in self.scenario_results if s.status == ScenarioStatus.FAILED)

    @property
    def pass_rate(self) -> float:
        """Taxa de sucesso (0-100)."""
        if self.total_scenarios == 0:
            return 0.0
        return (self.passed_scenarios / self.total_scenarios) * 100

    def get_result_by_scenario_id(self, scenario_id: str) -> Optional[ScenarioResult]:
        """Busca resultado por ID do cenario."""
        for r in self.scenario_results:
            if r.scenario_id == scenario_id:
                return r
        return None

    def to_dict(self) -> dict:
        """Converte para dicionario."""
        return {
            "executor": self.executor,
            "timestamp": self.timestamp.isoformat(),
            "base_url": self.base_url,
            "scenario_results": [r.to_dict() for r in self.scenario_results],
            "total_duration_ms": self.total_duration_ms,
            "metrics": {
                "total_scenarios": self.total_scenarios,
                "passed_scenarios": self.passed_scenarios,
                "failed_scenarios": self.failed_scenarios,
                "pass_rate": self.pass_rate,
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionResult":
        """Cria a partir de dicionario."""
        return cls(
            executor=data["executor"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            base_url=data["base_url"],
            scenario_results=[ScenarioResult.from_dict(r) for r in data.get("scenario_results", [])],
            total_duration_ms=data.get("total_duration_ms", 0),
        )


@dataclass
class Discrepancy:
    """
    Uma discrepancia entre resultados de diferentes executores.
    """
    scenario_id: str
    scenario_name: str
    browseruse_status: Optional[str] = None
    playwright_status: Optional[str] = None
    description: str = ""

    def to_dict(self) -> dict:
        """Converte para dicionario."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "browseruse_status": self.browseruse_status,
            "playwright_status": self.playwright_status,
            "description": self.description,
        }


@dataclass
class ConsolidatedReport:
    """
    Relatorio consolidado de todos os executores.

    Combina resultados do Browser-Use e Playwright,
    identificando discrepancias e gerando metricas finais.
    """
    timestamp: datetime
    base_url: str
    app_name: str

    # Resultados por executor
    browseruse_result: Optional[ExecutionResult] = None
    playwright_result: Optional[ExecutionResult] = None

    # Analise comparativa
    discrepancies: list[Discrepancy] = field(default_factory=list)

    # Metricas consolidadas
    total_scenarios_tested: int = 0
    scenarios_passed_both: int = 0
    scenarios_failed_both: int = 0
    scenarios_with_discrepancy: int = 0

    # UX insights (do Browser-Use)
    ux_insights: list[str] = field(default_factory=list)

    @property
    def overall_pass_rate(self) -> float:
        """Taxa de sucesso considerando ambos executores."""
        if self.total_scenarios_tested == 0:
            return 0.0
        return (self.scenarios_passed_both / self.total_scenarios_tested) * 100

    @property
    def browseruse_pass_rate(self) -> float:
        """Taxa de sucesso do Browser-Use."""
        if self.browseruse_result:
            return self.browseruse_result.pass_rate
        return 0.0

    @property
    def playwright_pass_rate(self) -> float:
        """Taxa de sucesso do Playwright."""
        if self.playwright_result:
            return self.playwright_result.pass_rate
        return 0.0

    def to_dict(self) -> dict:
        """Converte para dicionario."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "base_url": self.base_url,
            "app_name": self.app_name,
            "browseruse_result": self.browseruse_result.to_dict() if self.browseruse_result else None,
            "playwright_result": self.playwright_result.to_dict() if self.playwright_result else None,
            "discrepancies": [d.to_dict() for d in self.discrepancies],
            "metrics": {
                "total_scenarios_tested": self.total_scenarios_tested,
                "scenarios_passed_both": self.scenarios_passed_both,
                "scenarios_failed_both": self.scenarios_failed_both,
                "scenarios_with_discrepancy": self.scenarios_with_discrepancy,
                "overall_pass_rate": self.overall_pass_rate,
                "browseruse_pass_rate": self.browseruse_pass_rate,
                "playwright_pass_rate": self.playwright_pass_rate,
            },
            "ux_insights": self.ux_insights,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConsolidatedReport":
        """Cria a partir de dicionario."""
        bu_result = None
        pw_result = None

        if data.get("browseruse_result"):
            bu_result = ExecutionResult.from_dict(data["browseruse_result"])
        if data.get("playwright_result"):
            pw_result = ExecutionResult.from_dict(data["playwright_result"])

        metrics = data.get("metrics", {})

        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            base_url=data["base_url"],
            app_name=data["app_name"],
            browseruse_result=bu_result,
            playwright_result=pw_result,
            discrepancies=[],  # Simplified for now
            total_scenarios_tested=metrics.get("total_scenarios_tested", 0),
            scenarios_passed_both=metrics.get("scenarios_passed_both", 0),
            scenarios_failed_both=metrics.get("scenarios_failed_both", 0),
            scenarios_with_discrepancy=metrics.get("scenarios_with_discrepancy", 0),
            ux_insights=data.get("ux_insights", []),
        )

    def save(self, filepath: str) -> None:
        """Salva relatorio em arquivo JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "ConsolidatedReport":
        """Carrega relatorio de arquivo JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def export_html(self, filepath: str) -> None:
        """Exporta relatorio como HTML."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Test Report - {self.app_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .discrepancy {{ background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 8px; margin: 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Test Report - {self.app_name}</h1>
        <p><strong>URL:</strong> {self.base_url}</p>
        <p><strong>Data:</strong> {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{self.total_scenarios_tested}</div>
                <div class="metric-label">Total Scenarios</div>
            </div>
            <div class="metric">
                <div class="metric-value passed">{self.scenarios_passed_both}</div>
                <div class="metric-label">Passed (Both)</div>
            </div>
            <div class="metric">
                <div class="metric-value failed">{self.scenarios_failed_both}</div>
                <div class="metric-label">Failed (Both)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.overall_pass_rate:.1f}%</div>
                <div class="metric-label">Pass Rate</div>
            </div>
        </div>

        <h2>Executor Comparison</h2>
        <table>
            <tr>
                <th>Executor</th>
                <th>Total</th>
                <th>Passed</th>
                <th>Failed</th>
                <th>Pass Rate</th>
            </tr>
            <tr>
                <td>Browser-Use</td>
                <td>{self.browseruse_result.total_scenarios if self.browseruse_result else 0}</td>
                <td class="passed">{self.browseruse_result.passed_scenarios if self.browseruse_result else 0}</td>
                <td class="failed">{self.browseruse_result.failed_scenarios if self.browseruse_result else 0}</td>
                <td>{self.browseruse_pass_rate:.1f}%</td>
            </tr>
            <tr>
                <td>Playwright</td>
                <td>{self.playwright_result.total_scenarios if self.playwright_result else 0}</td>
                <td class="passed">{self.playwright_result.passed_scenarios if self.playwright_result else 0}</td>
                <td class="failed">{self.playwright_result.failed_scenarios if self.playwright_result else 0}</td>
                <td>{self.playwright_pass_rate:.1f}%</td>
            </tr>
        </table>

        {"<h2>Discrepancies</h2>" + "".join([f'<div class="discrepancy"><strong>{d.scenario_name}</strong><br>Browser-Use: {d.browseruse_status} | Playwright: {d.playwright_status}<br>{d.description}</div>' for d in self.discrepancies]) if self.discrepancies else ""}

        {"<h2>UX Insights</h2><ul>" + "".join([f"<li>{insight}</li>" for insight in self.ux_insights]) + "</ul>" if self.ux_insights else ""}
    </div>
</body>
</html>"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
