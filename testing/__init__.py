"""
Dumont Core - Testing Module

Sistema de testes automatizados com IA para aplicacoes web.
Usa exploracao colaborativa entre Browser-Use e Playwright para
descoberta e execucao de testes.

## Arquitetura

FASE 1: EXPLORACAO COLABORATIVA
- BrowserUseExplorer: Foco em UX, fluxos de usuario, edge cases
- PlaywrightExplorer: Foco em DOM, seletores, acessibilidade, APIs
- DiscoveryMerger: Consolida descobertas em plano unificado

FASE 2: EXECUCAO DE TESTES
- BrowserUseExecutor: Execucao adaptativa com julgamentos UX
- PlaywrightExecutor: Execucao deterministica para CI/CD
- ResultsConsolidator: Compara resultados e gera relatorio

## Uso Rapido

    from dumont_core.testing import UnifiedTestRunner

    runner = UnifiedTestRunner(
        base_url="http://localhost:80",
        output_dir="/tmp/test_results"
    )

    # Executa exploracao + testes
    results = await runner.run_full()
    print(f"Taxa de sucesso: {results.overall_pass_rate}%")

## Classes Principais

- UnifiedTestRunner: Orquestrador principal (recomendado)
- UIAnalyzer: Agente legado (compatibilidade)
"""

# Orquestrador principal
from .unified_runner import UnifiedTestRunner

# Modelos de dados
from .models import (
    # Classes
    Feature,
    UserFlow,
    DiscoveryResult,
    TestStep,
    TestScenario,
    UnifiedTestPlan,
    StepResult,
    ScenarioResult,
    ExecutionResult,
    ConsolidatedReport,
    Discrepancy,
    # Enums
    FeaturePriority,
    ElementType,
    ActionType,
    StepStatus,
    ScenarioStatus,
)

# Exploradores
from .exploration import (
    BaseExplorer,
    BrowserUseExplorer,
    PlaywrightExplorer,
    DiscoveryMerger,
)

# Executores
from .execution import (
    BaseExecutor,
    BrowserUseExecutor,
    PlaywrightExecutor,
    ResultsConsolidator,
)

# Classe legada (compatibilidade)
from .ui_analyzer import (
    UIAnalyzer,
    TestReport,
    TestResult,
    TestIssue,
    TestSeverity,
    TestCategory,
)

# Utilitarios
from .utils import (
    get_logger,
    retry_async,
    retry_sync,
    ValidationError,
    validate_url,
    validate_not_empty,
    validate_positive,
    validate_in_range,
    validate_enum,
    run_with_timeout,
    gather_with_errors,
    TimingContext,
    AsyncTimingContext,
    sanitize_filename,
    truncate_string,
)

__all__ = [
    # Orquestrador principal
    "UnifiedTestRunner",

    # Modelos de dados - Classes
    "Feature",
    "UserFlow",
    "DiscoveryResult",
    "TestStep",
    "TestScenario",
    "UnifiedTestPlan",
    "StepResult",
    "ScenarioResult",
    "ExecutionResult",
    "ConsolidatedReport",
    "Discrepancy",

    # Modelos de dados - Enums
    "FeaturePriority",
    "ElementType",
    "ActionType",
    "StepStatus",
    "ScenarioStatus",

    # Exploradores
    "BaseExplorer",
    "BrowserUseExplorer",
    "PlaywrightExplorer",
    "DiscoveryMerger",

    # Executores
    "BaseExecutor",
    "BrowserUseExecutor",
    "PlaywrightExecutor",
    "ResultsConsolidator",

    # Compatibilidade (legado)
    "UIAnalyzer",
    "TestReport",
    "TestResult",
    "TestIssue",
    "TestSeverity",
    "TestCategory",

    # Utilitarios
    "get_logger",
    "retry_async",
    "retry_sync",
    "ValidationError",
    "validate_url",
    "validate_not_empty",
    "validate_positive",
    "validate_in_range",
    "validate_enum",
    "run_with_timeout",
    "gather_with_errors",
    "TimingContext",
    "AsyncTimingContext",
    "sanitize_filename",
    "truncate_string",
]
