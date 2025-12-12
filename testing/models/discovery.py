"""
Modelos de dados para a fase de descoberta/exploracao.

Contem as estruturas usadas pelos exploradores (Browser-Use e Playwright)
para registrar suas descobertas antes do merge.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
import json


class FeaturePriority(str, Enum):
    """Prioridade de uma feature para testes."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ElementType(str, Enum):
    """Tipos de elementos UI detectados."""
    BUTTON = "button"
    INPUT = "input"
    LINK = "link"
    TEXT = "text"
    IMAGE = "image"
    FORM = "form"
    LIST = "list"
    TABLE = "table"
    MODAL = "modal"
    MENU = "menu"
    TAB = "tab"
    PANEL = "panel"
    OTHER = "other"


@dataclass
class Feature:
    """
    Uma feature/funcionalidade descoberta na aplicacao.

    Pode ser descoberta por Browser-Use (mais descritivo) ou
    Playwright (mais tecnico com seletores precisos).
    """
    id: str
    name: str
    element_type: ElementType
    priority: FeaturePriority = FeaturePriority.MEDIUM

    # Do Playwright (tecnico)
    selector: Optional[str] = None
    accessible_name: Optional[str] = None
    aria_role: Optional[str] = None

    # Do Browser-Use (descritivo)
    description: Optional[str] = None
    ux_notes: Optional[str] = None

    # Metadados
    source: str = "unknown"  # "browseruse", "playwright", "merged"
    location: Optional[str] = None  # "header", "sidebar", "main", etc.

    def __post_init__(self):
        """Valida campos apos inicializacao."""
        if not self.id or not self.id.strip():
            raise ValueError("Feature.id nao pode ser vazio")
        if not self.name or not self.name.strip():
            raise ValueError("Feature.name nao pode ser vazio")
        if isinstance(self.element_type, str):
            self.element_type = ElementType(self.element_type)
        if isinstance(self.priority, str):
            self.priority = FeaturePriority(self.priority)

    def to_dict(self) -> dict:
        """Converte para dicionario."""
        return {
            "id": self.id,
            "name": self.name,
            "element_type": self.element_type.value,
            "priority": self.priority.value,
            "selector": self.selector,
            "accessible_name": self.accessible_name,
            "aria_role": self.aria_role,
            "description": self.description,
            "ux_notes": self.ux_notes,
            "source": self.source,
            "location": self.location,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Feature":
        """Cria a partir de dicionario."""
        return cls(
            id=data["id"],
            name=data["name"],
            element_type=ElementType(data.get("element_type", "other")),
            priority=FeaturePriority(data.get("priority", "medium")),
            selector=data.get("selector"),
            accessible_name=data.get("accessible_name"),
            aria_role=data.get("aria_role"),
            description=data.get("description"),
            ux_notes=data.get("ux_notes"),
            source=data.get("source", "unknown"),
            location=data.get("location"),
        )


@dataclass
class UserFlow:
    """
    Um fluxo de usuario observado durante a exploracao.

    Descreve uma sequencia de acoes que um usuario faria
    para completar uma tarefa.
    """
    name: str
    steps: list[str]
    observed_behavior: str
    ux_notes: Optional[str] = None

    # Features envolvidas neste fluxo
    feature_ids: list[str] = field(default_factory=list)

    # Metadados
    source: str = "unknown"
    complexity: str = "simple"  # "simple", "medium", "complex"

    def to_dict(self) -> dict:
        """Converte para dicionario."""
        return {
            "name": self.name,
            "steps": self.steps,
            "observed_behavior": self.observed_behavior,
            "ux_notes": self.ux_notes,
            "feature_ids": self.feature_ids,
            "source": self.source,
            "complexity": self.complexity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserFlow":
        """Cria a partir de dicionario."""
        return cls(
            name=data["name"],
            steps=data["steps"],
            observed_behavior=data["observed_behavior"],
            ux_notes=data.get("ux_notes"),
            feature_ids=data.get("feature_ids", []),
            source=data.get("source", "unknown"),
            complexity=data.get("complexity", "simple"),
        )


@dataclass
class DiscoveryResult:
    """
    Resultado da exploracao de um explorador (Browser-Use ou Playwright).

    Contem todas as descobertas feitas durante a fase de exploracao,
    incluindo features, fluxos, problemas de acessibilidade e edge cases.
    """
    source: str  # "browseruse" ou "playwright"
    timestamp: datetime
    base_url: str

    # Descobertas principais
    features: list[Feature] = field(default_factory=list)
    user_flows: list[UserFlow] = field(default_factory=list)

    # Problemas encontrados
    accessibility_issues: list[str] = field(default_factory=list)
    edge_cases: list[str] = field(default_factory=list)

    # APIs detectadas (Playwright)
    api_endpoints: list[dict] = field(default_factory=list)

    # Dados brutos da exploracao
    raw_data: dict = field(default_factory=dict)

    # Screenshots capturados
    screenshots: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Converte para dicionario."""
        return {
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "base_url": self.base_url,
            "features": [f.to_dict() for f in self.features],
            "user_flows": [f.to_dict() for f in self.user_flows],
            "accessibility_issues": self.accessibility_issues,
            "edge_cases": self.edge_cases,
            "api_endpoints": self.api_endpoints,
            "raw_data": self.raw_data,
            "screenshots": self.screenshots,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DiscoveryResult":
        """Cria a partir de dicionario."""
        return cls(
            source=data["source"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            base_url=data["base_url"],
            features=[Feature.from_dict(f) for f in data.get("features", [])],
            user_flows=[UserFlow.from_dict(f) for f in data.get("user_flows", [])],
            accessibility_issues=data.get("accessibility_issues", []),
            edge_cases=data.get("edge_cases", []),
            api_endpoints=data.get("api_endpoints", []),
            raw_data=data.get("raw_data", {}),
            screenshots=data.get("screenshots", []),
        )

    def save(self, filepath: str) -> None:
        """Salva resultado em arquivo JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "DiscoveryResult":
        """Carrega resultado de arquivo JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
