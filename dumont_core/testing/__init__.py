"""
Dumont Core - Testing Module

Agente de teste automatizado com IA para aplicações web.
Usa Browser-Use + LLMs com visão para testar como um usuário real.

Funcionalidades:
- Testes funcionais (navegação, formulários, botões)
- Testes de performance (tempo de carregamento, responsividade)
- Testes de acessibilidade (WCAG compliance)
- Análise visual/UX
- Detecção automática de bugs
- Sugestões de melhoria com IA

Uso básico:
    from dumont_core.testing import UIAnalyzer

    analyzer = UIAnalyzer(base_url="http://localhost:8080")
    report = await analyzer.run_full_test()

    # Ou testes específicos:
    result = await analyzer.test_functionality()
    result = await analyzer.test_performance()
    result = await analyzer.test_accessibility()

Classes exportadas:
- UIAnalyzer: Agente principal de testes
- TestReport: Relatório completo de testes
- TestResult: Resultado de um teste individual
- TestIssue: Issue/problema encontrado
- TestSeverity: Enum de severidades (CRITICAL, HIGH, MEDIUM, LOW, INFO)
- TestCategory: Enum de categorias (FUNCTIONAL, PERFORMANCE, etc)
"""

from dumont_core.testing.ui_analyzer import (
    UIAnalyzer,
    TestReport,
    TestResult,
    TestIssue,
    TestSeverity,
    TestCategory,
)

__all__ = [
    "UIAnalyzer",
    "TestReport",
    "TestResult",
    "TestIssue",
    "TestSeverity",
    "TestCategory",
]
