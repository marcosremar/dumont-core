"""
Executor baseado em Playwright.

Executa cenarios de teste usando Playwright puro,
permitindo execucao deterministica e rapida.
"""

import time
from datetime import datetime
from typing import Optional
from pathlib import Path

from .base_executor import BaseExecutor
from ..models.unified_plan import (
    UnifiedTestPlan,
    TestScenario,
    TestStep,
    ActionType,
)
from ..models.test_result import (
    ExecutionResult,
    ScenarioResult,
    StepResult,
    StepStatus,
    ScenarioStatus,
)


class PlaywrightExecutor(BaseExecutor):
    """
    Executor que usa Playwright diretamente.

    Caracteristicas:
    - Execucao deterministica e rapida
    - Ideal para CI/CD
    - Metricas precisas de tempo
    - Usa seletores definidos no plano
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        headless: bool = True,
        timeout: int = 60,
    ):
        super().__init__(output_dir, headless, timeout)
        self._browser = None
        self._page = None

    @property
    def name(self) -> str:
        return "playwright"

    async def execute(self, plan: UnifiedTestPlan) -> ExecutionResult:
        """Executa todos os cenarios do plano."""
        from playwright.async_api import async_playwright

        # Configura output
        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        scenario_results = []
        total_start = time.time()

        async with async_playwright() as p:
            self._browser = await p.chromium.launch(headless=self.headless)
            context = await self._browser.new_context()

            # Filtra cenarios que este executor pode testar
            scenarios = [s for s in plan.test_scenarios if s.playwright_can_test]

            for scenario in scenarios:
                # Cria nova pagina para cada cenario
                self._page = await context.new_page()
                self._page.set_default_timeout(self.timeout * 1000)

                try:
                    result = await self.execute_scenario(scenario, plan.base_url)
                    scenario_results.append(result)
                finally:
                    await self._page.close()

            await self._browser.close()

        total_duration = int((time.time() - total_start) * 1000)

        return ExecutionResult(
            executor=self.name,
            timestamp=datetime.now(),
            base_url=plan.base_url,
            scenario_results=scenario_results,
            total_duration_ms=total_duration,
        )

    async def execute_scenario(self, scenario: TestScenario, base_url: str) -> ScenarioResult:
        """Executa um cenario usando Playwright."""
        start_time = datetime.now()
        step_results = []
        screenshots = []
        overall_error = None

        for i, step in enumerate(scenario.steps):
            step_start = time.time()

            try:
                await self._execute_step(step, base_url)

                step_duration = int((time.time() - step_start) * 1000)
                step_results.append(StepResult(
                    step_index=i,
                    action=step.action.value,
                    status=StepStatus.PASSED,
                    duration_ms=step_duration,
                    message=step.description,
                ))

            except Exception as e:
                step_duration = int((time.time() - step_start) * 1000)
                step_results.append(StepResult(
                    step_index=i,
                    action=step.action.value,
                    status=StepStatus.FAILED,
                    duration_ms=step_duration,
                    error=str(e),
                ))

                overall_error = str(e)

                # Para execucao no primeiro erro
                # Marca passos restantes como skipped
                for j in range(i + 1, len(scenario.steps)):
                    remaining_step = scenario.steps[j]
                    step_results.append(StepResult(
                        step_index=j,
                        action=remaining_step.action.value,
                        status=StepStatus.SKIPPED,
                        message="Pulado devido a erro anterior",
                    ))
                break

        end_time = datetime.now()
        duration = int((end_time - start_time).total_seconds() * 1000)

        # Determina status geral
        if overall_error:
            status = ScenarioStatus.FAILED
        elif all(r.status == StepStatus.PASSED for r in step_results):
            status = ScenarioStatus.PASSED
        else:
            status = ScenarioStatus.FAILED

        return ScenarioResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            status=status,
            executor=self.name,
            step_results=step_results,
            duration_ms=duration,
            start_time=start_time,
            end_time=end_time,
            error=overall_error,
            screenshots=screenshots,
        )

    async def _execute_step(self, step: TestStep, base_url: str):
        """Executa um passo individual."""
        action = step.action
        target = step.target
        value = step.value
        timeout = step.timeout * 1000  # Converte para ms

        if action == ActionType.NAVIGATE:
            url = target if target.startswith("http") else base_url
            await self._page.goto(url, wait_until="networkidle", timeout=timeout)

        elif action == ActionType.CLICK:
            element = await self._find_element(target, timeout)
            await element.click()

        elif action == ActionType.TYPE:
            element = await self._find_element(target, timeout)
            await element.fill(value or "")

        elif action == ActionType.WAIT:
            if target == "networkidle":
                await self._page.wait_for_load_state("networkidle", timeout=timeout)
            else:
                # Espera por selector
                await self._page.wait_for_selector(target, timeout=timeout)

        elif action == ActionType.VERIFY:
            element = await self._find_element(target, timeout)

            if step.condition:
                if step.condition == "visible":
                    is_visible = await element.is_visible()
                    if not is_visible:
                        raise AssertionError(f"Elemento {target} nao esta visivel")
                elif step.condition.startswith("value =="):
                    expected = step.condition.split("==")[1].strip().strip("'\"")
                    actual = await element.input_value()
                    if actual != expected:
                        raise AssertionError(f"Valor esperado '{expected}', encontrado '{actual}'")
                elif step.condition.startswith("text =="):
                    expected = step.condition.split("==")[1].strip().strip("'\"")
                    actual = await element.text_content()
                    if actual != expected:
                        raise AssertionError(f"Texto esperado '{expected}', encontrado '{actual}'")

        elif action == ActionType.SCROLL:
            if target == "bottom":
                await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            elif target == "top":
                await self._page.evaluate("window.scrollTo(0, 0)")
            else:
                element = await self._find_element(target, timeout)
                await element.scroll_into_view_if_needed()

        elif action == ActionType.HOVER:
            element = await self._find_element(target, timeout)
            await element.hover()

        elif action == ActionType.SELECT:
            element = await self._find_element(target, timeout)
            await element.select_option(value=value)

        elif action == ActionType.SCREENSHOT:
            if self.output_dir:
                screenshot_path = str(Path(self.output_dir) / f"{target}.png")
                await self._page.screenshot(path=screenshot_path, full_page=True)

        elif action == ActionType.CUSTOM:
            # Para acoes customizadas, tenta interpretar o target como descricao
            # Por enquanto, apenas loga
            pass

    async def _find_element(self, target: str, timeout: int):
        """Encontra elemento pelo seletor ou texto."""
        # Se parece com seletor CSS
        if target.startswith(("#", ".", "[")) or "::" in target:
            return await self._page.wait_for_selector(target, timeout=timeout)

        # Se tem "has-text", ja e um seletor valido
        if ":has-text" in target or ":text" in target:
            return await self._page.wait_for_selector(target, timeout=timeout)

        # Tenta encontrar por texto
        try:
            selector = f"text={target}"
            return await self._page.wait_for_selector(selector, timeout=timeout)
        except Exception:
            pass

        # Tenta encontrar por aria-label
        try:
            selector = f"[aria-label='{target}']"
            return await self._page.wait_for_selector(selector, timeout=timeout)
        except Exception:
            pass

        # Tenta encontrar por placeholder
        try:
            selector = f"[placeholder='{target}']"
            return await self._page.wait_for_selector(selector, timeout=timeout)
        except Exception:
            pass

        raise Exception(f"Elemento nao encontrado: {target}")

    async def cleanup(self):
        """Fecha browser."""
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        self._page = None
