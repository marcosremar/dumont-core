"""
Modelos de dados para o plano de testes unificado.

O plano unificado e gerado pelo Discovery Merger apos combinar
as descobertas de todos os exploradores.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
import json

from dumont_core.testing.models.discovery import (
    Feature,
    DiscoveryResult,
    FeaturePriority,
)


class ActionType(str, Enum):
    """Tipos de acoes em um step de teste."""
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    WAIT = "wait"
    VERIFY = "verify"
    SCROLL = "scroll"
    HOVER = "hover"
    SELECT = "select"
    SCREENSHOT = "screenshot"
    CUSTOM = "custom"


@dataclass
class TestStep:
    """
    Um passo individual em um cenario de teste.

    Representa uma acao atomica como clicar, digitar, verificar, etc.
    """
    action: ActionType
    target: str  # feature_id, selector ou descricao
    value: Optional[str] = None  # valor para type, URL para navigate, etc.
    condition: Optional[str] = None  # condicao para verify
    timeout: int = 30  # timeout em segundos

    # Metadados
    description: Optional[str] = None

    def __post_init__(self):
        """Valida campos apos inicializacao."""
        if isinstance(self.action, str):
            self.action = ActionType(self.action)
        if not self.target or not self.target.strip():
            raise ValueError("TestStep.target nao pode ser vazio")
        if self.timeout < 0:
            raise ValueError(f"TestStep.timeout deve ser >= 0, recebeu {self.timeout}")

    def to_dict(self) -> dict:
        """Converte para dicionario."""
        return {
            "action": self.action.value,
            "target": self.target,
            "value": self.value,
            "condition": self.condition,
            "timeout": self.timeout,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TestStep":
        """Cria a partir de dicionario."""
        return cls(
            action=ActionType(data["action"]),
            target=data["target"],
            value=data.get("value"),
            condition=data.get("condition"),
            timeout=data.get("timeout", 30),
            description=data.get("description"),
        )


@dataclass
class TestScenario:
    """
    Um cenario de teste completo.

    Agrupa varios TestSteps para testar uma funcionalidade ou fluxo.
    """
    id: str
    name: str
    steps: list[TestStep]
    expected_result: str
    priority: FeaturePriority = FeaturePriority.MEDIUM

    # Capacidades de execucao
    playwright_can_test: bool = True
    browseruse_can_test: bool = True
    ux_verification_needed: bool = False

    # Features relacionadas
    feature_ids: list[str] = field(default_factory=list)

    # Metadados
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Valida campos apos inicializacao."""
        if not self.id or not self.id.strip():
            raise ValueError("TestScenario.id nao pode ser vazio")
        if not self.name or not self.name.strip():
            raise ValueError("TestScenario.name nao pode ser vazio")
        if not self.steps:
            raise ValueError("TestScenario.steps nao pode ser vazio")
        if not self.expected_result or not self.expected_result.strip():
            raise ValueError("TestScenario.expected_result nao pode ser vazio")
        if isinstance(self.priority, str):
            self.priority = FeaturePriority(self.priority)

    def to_dict(self) -> dict:
        """Converte para dicionario."""
        return {
            "id": self.id,
            "name": self.name,
            "steps": [s.to_dict() for s in self.steps],
            "expected_result": self.expected_result,
            "priority": self.priority.value,
            "playwright_can_test": self.playwright_can_test,
            "browseruse_can_test": self.browseruse_can_test,
            "ux_verification_needed": self.ux_verification_needed,
            "feature_ids": self.feature_ids,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TestScenario":
        """Cria a partir de dicionario."""
        return cls(
            id=data["id"],
            name=data["name"],
            steps=[TestStep.from_dict(s) for s in data["steps"]],
            expected_result=data["expected_result"],
            priority=FeaturePriority(data.get("priority", "medium")),
            playwright_can_test=data.get("playwright_can_test", True),
            browseruse_can_test=data.get("browseruse_can_test", True),
            ux_verification_needed=data.get("ux_verification_needed", False),
            feature_ids=data.get("feature_ids", []),
            description=data.get("description"),
            tags=data.get("tags", []),
        )


@dataclass
class UnifiedTestPlan:
    """
    Plano de testes unificado.

    Contem todas as features descobertas e cenarios de teste gerados
    a partir das descobertas dos exploradores.
    """
    app_name: str
    base_url: str
    exploration_timestamp: datetime

    # Features consolidadas
    features: list[Feature] = field(default_factory=list)

    # Cenarios de teste
    test_scenarios: list[TestScenario] = field(default_factory=list)

    # Descobertas originais (para referencia)
    browseruse_discovery: Optional[DiscoveryResult] = None
    playwright_discovery: Optional[DiscoveryResult] = None

    # Insights combinados
    insights: dict = field(default_factory=dict)

    # Metadados
    version: str = "1.0"

    @property
    def total_scenarios(self) -> int:
        """Total de cenarios de teste."""
        return len(self.test_scenarios)

    @property
    def critical_scenarios(self) -> list[TestScenario]:
        """Cenarios com prioridade critica."""
        return [s for s in self.test_scenarios if s.priority == FeaturePriority.CRITICAL]

    @property
    def playwright_scenarios(self) -> list[TestScenario]:
        """Cenarios que podem ser testados com Playwright."""
        return [s for s in self.test_scenarios if s.playwright_can_test]

    @property
    def browseruse_scenarios(self) -> list[TestScenario]:
        """Cenarios que podem ser testados com Browser-Use."""
        return [s for s in self.test_scenarios if s.browseruse_can_test]

    def get_feature_by_id(self, feature_id: str) -> Optional[Feature]:
        """Busca feature por ID."""
        for f in self.features:
            if f.id == feature_id:
                return f
        return None

    def get_scenario_by_id(self, scenario_id: str) -> Optional[TestScenario]:
        """Busca cenario por ID."""
        for s in self.test_scenarios:
            if s.id == scenario_id:
                return s
        return None

    def to_dict(self) -> dict:
        """Converte para dicionario."""
        return {
            "app_name": self.app_name,
            "base_url": self.base_url,
            "exploration_timestamp": self.exploration_timestamp.isoformat(),
            "features": [f.to_dict() for f in self.features],
            "test_scenarios": [s.to_dict() for s in self.test_scenarios],
            "browseruse_discovery": self.browseruse_discovery.to_dict() if self.browseruse_discovery else None,
            "playwright_discovery": self.playwright_discovery.to_dict() if self.playwright_discovery else None,
            "insights": self.insights,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UnifiedTestPlan":
        """Cria a partir de dicionario."""
        bu_disc = None
        pw_disc = None

        if data.get("browseruse_discovery"):
            bu_disc = DiscoveryResult.from_dict(data["browseruse_discovery"])
        if data.get("playwright_discovery"):
            pw_disc = DiscoveryResult.from_dict(data["playwright_discovery"])

        return cls(
            app_name=data["app_name"],
            base_url=data["base_url"],
            exploration_timestamp=datetime.fromisoformat(data["exploration_timestamp"]),
            features=[Feature.from_dict(f) for f in data.get("features", [])],
            test_scenarios=[TestScenario.from_dict(s) for s in data.get("test_scenarios", [])],
            browseruse_discovery=bu_disc,
            playwright_discovery=pw_disc,
            insights=data.get("insights", {}),
            version=data.get("version", "1.0"),
        )

    def save(self, filepath: str) -> None:
        """Salva plano em arquivo JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "UnifiedTestPlan":
        """Carrega plano de arquivo JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def export_markdown(self, filepath: str) -> None:
        """Exporta plano como Markdown legivel."""
        lines = [
            f"# Plano de Testes - {self.app_name}",
            f"",
            f"**URL Base:** {self.base_url}",
            f"**Data da Exploracao:** {self.exploration_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total de Cenarios:** {self.total_scenarios}",
            f"",
            "## Features Descobertas",
            f"",
        ]

        for f in self.features:
            lines.append(f"### {f.name}")
            lines.append(f"- **ID:** {f.id}")
            lines.append(f"- **Tipo:** {f.element_type.value}")
            lines.append(f"- **Prioridade:** {f.priority.value}")
            if f.description:
                lines.append(f"- **Descricao:** {f.description}")
            if f.selector:
                lines.append(f"- **Seletor:** `{f.selector}`")
            lines.append("")

        lines.append("## Cenarios de Teste")
        lines.append("")

        for s in self.test_scenarios:
            lines.append(f"### {s.name}")
            lines.append(f"- **ID:** {s.id}")
            lines.append(f"- **Prioridade:** {s.priority.value}")
            lines.append(f"- **Resultado Esperado:** {s.expected_result}")
            lines.append("")
            lines.append("**Passos:**")
            for i, step in enumerate(s.steps, 1):
                desc = step.description or f"{step.action.value} em {step.target}"
                lines.append(f"{i}. {desc}")
            lines.append("")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
