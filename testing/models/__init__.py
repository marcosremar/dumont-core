"""
Modelos de dados para o sistema de testes colaborativos.

Exporta:
- Feature, UserFlow, DiscoveryResult (discovery.py)
- TestStep, TestScenario, UnifiedTestPlan (unified_plan.py)
- ExecutionResult, ConsolidatedReport (test_result.py)
- Enums: FeaturePriority, ElementType, ActionType, StepStatus, ScenarioStatus
"""

from dumont_core.testing.models.discovery import (
    Feature,
    UserFlow,
    DiscoveryResult,
    FeaturePriority,
    ElementType,
)

from dumont_core.testing.models.unified_plan import (
    TestStep,
    TestScenario,
    UnifiedTestPlan,
    ActionType,
)

from dumont_core.testing.models.test_result import (
    StepResult,
    ScenarioResult,
    ExecutionResult,
    ConsolidatedReport,
    Discrepancy,
    StepStatus,
    ScenarioStatus,
)

__all__ = [
    # Enums - Discovery
    "FeaturePriority",
    "ElementType",
    # Enums - Unified Plan
    "ActionType",
    # Enums - Test Results
    "StepStatus",
    "ScenarioStatus",
    # Discovery
    "Feature",
    "UserFlow",
    "DiscoveryResult",
    # Unified Plan
    "TestStep",
    "TestScenario",
    "UnifiedTestPlan",
    # Test Results
    "StepResult",
    "ScenarioResult",
    "ExecutionResult",
    "ConsolidatedReport",
    "Discrepancy",
]
