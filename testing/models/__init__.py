"""
Modelos de dados para o sistema de testes colaborativos.

Exporta:
- Feature, UserFlow, DiscoveryResult (discovery.py)
- TestStep, TestScenario, UnifiedTestPlan (unified_plan.py)
- ExecutionResult, ConsolidatedReport (test_result.py)
- Enums: FeaturePriority, ElementType, ActionType, StepStatus, ScenarioStatus
"""

from .discovery import (
    Feature,
    UserFlow,
    DiscoveryResult,
    FeaturePriority,
    ElementType,
)

from .unified_plan import (
    TestStep,
    TestScenario,
    UnifiedTestPlan,
    ActionType,
)

from .test_result import (
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
