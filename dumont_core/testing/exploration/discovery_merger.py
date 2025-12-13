"""
Merger de descobertas de diferentes exploradores.

Consolida os resultados de Browser-Use e Playwright em um
UnifiedTestPlan pronto para execucao.
"""

import uuid
from datetime import datetime
from typing import Optional

from dumont_core.testing.models.discovery import (
    DiscoveryResult,
    Feature,
    UserFlow,
    FeaturePriority,
)
from dumont_core.testing.models.unified_plan import (
    UnifiedTestPlan,
    TestScenario,
    TestStep,
    ActionType,
)


class DiscoveryMerger:
    """
    Combina descobertas de multiplos exploradores.

    Logica de merge:
    1. Features com mesmo nome/seletor sao combinadas
    2. Playwright fornece seletores, Browser-Use fornece descricoes
    3. Cenarios de teste sao gerados automaticamente
    4. Insights de ambos sao preservados
    """

    def __init__(
        self,
        app_name: str,
        base_url: str,
    ):
        self.app_name = app_name
        self.base_url = base_url

    def merge(
        self,
        browseruse_result: Optional[DiscoveryResult] = None,
        playwright_result: Optional[DiscoveryResult] = None,
    ) -> UnifiedTestPlan:
        """
        Combina resultados de exploracao em plano unificado.

        Args:
            browseruse_result: Resultado do Browser-Use Explorer
            playwright_result: Resultado do Playwright Explorer

        Returns:
            UnifiedTestPlan pronto para execucao
        """
        # Coleta todas as features
        all_features = []
        if browseruse_result:
            all_features.extend(browseruse_result.features)
        if playwright_result:
            all_features.extend(playwright_result.features)

        # Merge features (combina duplicatas)
        merged_features = self._merge_features(all_features)

        # Coleta todos os fluxos
        all_flows = []
        if browseruse_result:
            all_flows.extend(browseruse_result.user_flows)
        if playwright_result:
            all_flows.extend(playwright_result.user_flows)

        # Gera cenarios de teste
        test_scenarios = self._generate_test_scenarios(merged_features, all_flows)

        # Combina insights
        insights = self._combine_insights(browseruse_result, playwright_result)

        return UnifiedTestPlan(
            app_name=self.app_name,
            base_url=self.base_url,
            exploration_timestamp=datetime.now(),
            features=merged_features,
            test_scenarios=test_scenarios,
            browseruse_discovery=browseruse_result,
            playwright_discovery=playwright_result,
            insights=insights,
        )

    def _merge_features(self, features: list[Feature]) -> list[Feature]:
        """
        Combina features duplicadas.

        Estrategia:
        - Agrupa por nome similar
        - Prefere seletores do Playwright
        - Prefere descricoes do Browser-Use
        """
        merged = {}

        for feature in features:
            # Normaliza nome para comparacao
            key = self._normalize_name(feature.name)

            if key in merged:
                # Combina com feature existente
                existing = merged[key]
                merged[key] = self._combine_features(existing, feature)
            else:
                # Nova feature
                merged[key] = feature

        return list(merged.values())

    def _normalize_name(self, name: str) -> str:
        """Normaliza nome para comparacao."""
        return name.lower().strip().replace(" ", "_")[:30]

    def _combine_features(self, f1: Feature, f2: Feature) -> Feature:
        """Combina duas features em uma."""
        # Prefere dados do Playwright para tecnico
        selector = f1.selector or f2.selector
        accessible_name = f1.accessible_name or f2.accessible_name
        aria_role = f1.aria_role or f2.aria_role

        # Prefere dados do Browser-Use para descritivo
        description = None
        ux_notes = None
        if f1.source == "browseruse":
            description = f1.description
            ux_notes = f1.ux_notes
        elif f2.source == "browseruse":
            description = f2.description
            ux_notes = f2.ux_notes

        # Usa a maior prioridade
        priority = f1.priority if f1.priority.value < f2.priority.value else f2.priority

        return Feature(
            id=f"merged_{uuid.uuid4().hex[:8]}",
            name=f1.name,  # Usa primeiro nome
            element_type=f1.element_type,
            priority=priority,
            selector=selector,
            accessible_name=accessible_name,
            aria_role=aria_role,
            description=description,
            ux_notes=ux_notes,
            source="merged",
            location=f1.location or f2.location,
        )

    def _generate_test_scenarios(
        self,
        features: list[Feature],
        flows: list[UserFlow],
    ) -> list[TestScenario]:
        """
        Gera cenarios de teste a partir de features e fluxos.
        """
        scenarios = []

        # Cenario 1: Teste de cada feature critica/alta
        for feature in features:
            if feature.priority in [FeaturePriority.CRITICAL, FeaturePriority.HIGH]:
                scenario = self._create_feature_scenario(feature)
                if scenario:
                    scenarios.append(scenario)

        # Cenario 2: Teste de fluxos de usuario
        for flow in flows:
            scenario = self._create_flow_scenario(flow, features)
            if scenario:
                scenarios.append(scenario)

        # Cenario 3: Cenario de navegacao inicial
        scenarios.insert(0, TestScenario(
            id=f"ts_{uuid.uuid4().hex[:8]}",
            name="Carregamento Inicial",
            steps=[
                TestStep(
                    action=ActionType.NAVIGATE,
                    target=self.base_url,
                    description="Navegar para pagina inicial",
                ),
                TestStep(
                    action=ActionType.WAIT,
                    target="networkidle",
                    timeout=30,
                    description="Aguardar carregamento completo",
                ),
                TestStep(
                    action=ActionType.SCREENSHOT,
                    target="initial_state",
                    description="Capturar estado inicial",
                ),
            ],
            expected_result="Pagina carrega sem erros",
            priority=FeaturePriority.CRITICAL,
            playwright_can_test=True,
            browseruse_can_test=True,
            tags=["smoke", "initial"],
        ))

        return scenarios

    def _create_feature_scenario(self, feature: Feature) -> Optional[TestScenario]:
        """Cria cenario de teste para uma feature."""
        steps = []

        # Navega para pagina
        steps.append(TestStep(
            action=ActionType.NAVIGATE,
            target=self.base_url,
            description="Navegar para aplicacao",
        ))

        # Acao baseada no tipo de elemento
        if feature.element_type.value == "button":
            steps.append(TestStep(
                action=ActionType.CLICK,
                target=feature.selector or feature.name,
                description=f"Clicar em {feature.name}",
            ))
            steps.append(TestStep(
                action=ActionType.WAIT,
                target="networkidle",
                timeout=10,
                description="Aguardar resposta",
            ))
        elif feature.element_type.value == "input":
            steps.append(TestStep(
                action=ActionType.TYPE,
                target=feature.selector or feature.name,
                value="Texto de teste",
                description=f"Digitar em {feature.name}",
            ))
            steps.append(TestStep(
                action=ActionType.VERIFY,
                target=feature.selector or feature.name,
                condition="value == 'Texto de teste'",
                description="Verificar texto digitado",
            ))
        elif feature.element_type.value == "link":
            steps.append(TestStep(
                action=ActionType.CLICK,
                target=feature.selector or feature.name,
                description=f"Clicar em link {feature.name}",
            ))
            steps.append(TestStep(
                action=ActionType.WAIT,
                target="networkidle",
                timeout=10,
                description="Aguardar navegacao",
            ))
        else:
            # Elemento generico - apenas verifica visibilidade
            steps.append(TestStep(
                action=ActionType.VERIFY,
                target=feature.selector or feature.name,
                condition="visible",
                description=f"Verificar {feature.name} visivel",
            ))

        # Screenshot final
        steps.append(TestStep(
            action=ActionType.SCREENSHOT,
            target=f"feature_{self._normalize_name(feature.name)}",
            description="Capturar estado final",
        ))

        return TestScenario(
            id=f"ts_{uuid.uuid4().hex[:8]}",
            name=f"Teste: {feature.name}",
            steps=steps,
            expected_result=f"Feature {feature.name} funciona corretamente",
            priority=feature.priority,
            playwright_can_test=bool(feature.selector),
            browseruse_can_test=True,
            ux_verification_needed=feature.element_type.value in ["button", "form"],
            feature_ids=[feature.id],
            description=feature.description,
            tags=[feature.element_type.value],
        )

    def _create_flow_scenario(
        self,
        flow: UserFlow,
        features: list[Feature],
    ) -> Optional[TestScenario]:
        """Cria cenario de teste a partir de um fluxo de usuario."""
        if not flow.steps:
            return None

        # Mapeia features por ID para lookup
        feature_map = {f.id: f for f in features}

        steps = [
            TestStep(
                action=ActionType.NAVIGATE,
                target=self.base_url,
                description="Navegar para aplicacao",
            )
        ]

        # Converte passos do fluxo em TestSteps
        for i, step_desc in enumerate(flow.steps):
            # Tenta inferir acao do texto
            step_desc_lower = step_desc.lower()

            if "clicar" in step_desc_lower or "click" in step_desc_lower:
                action = ActionType.CLICK
            elif "digitar" in step_desc_lower or "type" in step_desc_lower or "preencher" in step_desc_lower:
                action = ActionType.TYPE
            elif "aguardar" in step_desc_lower or "wait" in step_desc_lower:
                action = ActionType.WAIT
            elif "verificar" in step_desc_lower or "verify" in step_desc_lower:
                action = ActionType.VERIFY
            else:
                action = ActionType.CUSTOM

            # Tenta encontrar feature relacionada
            target = step_desc
            for fid in flow.feature_ids:
                if fid in feature_map:
                    f = feature_map[fid]
                    if f.name.lower() in step_desc_lower:
                        target = f.selector or f.name
                        break

            steps.append(TestStep(
                action=action,
                target=target,
                description=step_desc,
                value="Texto de teste" if action == ActionType.TYPE else None,
            ))

        # Screenshot final
        steps.append(TestStep(
            action=ActionType.SCREENSHOT,
            target=f"flow_{self._normalize_name(flow.name)}",
            description="Capturar estado final do fluxo",
        ))

        return TestScenario(
            id=f"ts_{uuid.uuid4().hex[:8]}",
            name=f"Fluxo: {flow.name}",
            steps=steps,
            expected_result=flow.observed_behavior or f"Fluxo {flow.name} completo",
            priority=FeaturePriority.HIGH,
            playwright_can_test=True,
            browseruse_can_test=True,
            ux_verification_needed=True,
            feature_ids=flow.feature_ids,
            description=flow.ux_notes,
            tags=["flow", flow.complexity],
        )

    def _combine_insights(
        self,
        browseruse_result: Optional[DiscoveryResult],
        playwright_result: Optional[DiscoveryResult],
    ) -> dict:
        """Combina insights de ambos exploradores."""
        insights = {
            "accessibility_issues": [],
            "edge_cases": [],
            "api_endpoints": [],
            "ux_notes": [],
        }

        if browseruse_result:
            insights["accessibility_issues"].extend(browseruse_result.accessibility_issues)
            insights["edge_cases"].extend(browseruse_result.edge_cases)

            # Coleta UX notes das features
            for f in browseruse_result.features:
                if f.ux_notes:
                    insights["ux_notes"].append(f"{f.name}: {f.ux_notes}")

        if playwright_result:
            insights["accessibility_issues"].extend(playwright_result.accessibility_issues)
            insights["api_endpoints"].extend(playwright_result.api_endpoints)

        # Remove duplicatas
        insights["accessibility_issues"] = list(set(insights["accessibility_issues"]))
        insights["edge_cases"] = list(set(insights["edge_cases"]))

        return insights
