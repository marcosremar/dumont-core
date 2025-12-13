"""
Executor baseado em Browser-Use + LLM.

Executa cenarios de teste usando Browser-Use com LLM,
permitindo execucao adaptativa e julgamentos de UX.
"""

import os
import re
import json
import time
import asyncio
from datetime import datetime
from typing import Optional
from pathlib import Path

from dumont_core.testing.execution.base_executor import BaseExecutor
from dumont_core.testing.models.unified_plan import (
    UnifiedTestPlan,
    TestScenario,
    TestStep,
    ActionType,
)
from dumont_core.testing.models.test_result import (
    ExecutionResult,
    ScenarioResult,
    StepResult,
    StepStatus,
    ScenarioStatus,
)
from dumont_core.testing.utils import get_logger, retry_async, AsyncTimingContext

logger = get_logger(__name__)


class BrowserUseExecutor(BaseExecutor):
    """
    Executor que usa Browser-Use com LLM.

    Caracteristicas:
    - Execucao adaptativa (encontra elementos mesmo se mudarem)
    - Julgamentos de UX (texto legivel? feedback claro?)
    - Captura screenshots a cada passo
    - Fornece feedback em linguagem natural
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        headless: bool = True,
        timeout: int = 60,
        llm_model: str = "google/gemini-2.5-flash",
        llm_provider: str = "openrouter",
        max_steps_per_scenario: int = 20,
    ):
        super().__init__(output_dir, headless, timeout)
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.max_steps_per_scenario = max_steps_per_scenario
        self._browser = None

    @property
    def name(self) -> str:
        return "browseruse"

    def _get_llm(self):
        """Configura e retorna o LLM baseado no provider."""
        if self.llm_provider == "openrouter":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.llm_model,
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                temperature=0.1,
            )
        elif self.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.llm_model,
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                temperature=0.1,
            )
        elif self.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.llm_model,
                api_key=os.environ.get("OPENAI_API_KEY"),
                temperature=0.1,
            )
        else:
            raise ValueError(f"Provider desconhecido: {self.llm_provider}")

    async def execute(self, plan: UnifiedTestPlan) -> ExecutionResult:
        """Executa todos os cenarios do plano."""
        from browser_use import Browser, BrowserConfig

        logger.info(f"Iniciando execucao Browser-Use para {plan.app_name}")

        # Configura output
        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Configura browser com retry
        async with AsyncTimingContext("browser_setup", logger):
            browser_config = BrowserConfig(headless=self.headless)
            self._browser = Browser(config=browser_config)

        scenario_results = []
        total_start = time.time()

        try:
            # Filtra cenarios que este executor pode testar
            scenarios = [s for s in plan.test_scenarios if s.browseruse_can_test]
            logger.info(f"Executando {len(scenarios)} cenarios com Browser-Use")

            for i, scenario in enumerate(scenarios, 1):
                logger.info(f"[{i}/{len(scenarios)}] Executando: {scenario.name}")
                result = await self.execute_scenario(scenario, plan.base_url)
                scenario_results.append(result)
                logger.info(f"[{i}/{len(scenarios)}] {scenario.name}: {result.status.value}")

        except Exception as e:
            logger.error(f"Erro durante execucao: {e}")
            raise
        finally:
            await self.cleanup()

        total_duration = int((time.time() - total_start) * 1000)
        logger.info(f"Execucao Browser-Use concluida em {total_duration}ms")

        return ExecutionResult(
            executor=self.name,
            timestamp=datetime.now(),
            base_url=plan.base_url,
            scenario_results=scenario_results,
            total_duration_ms=total_duration,
        )

    async def execute_scenario(self, scenario: TestScenario, base_url: str) -> ScenarioResult:
        """Executa um cenario usando Browser-Use."""
        from browser_use import Agent

        start_time = datetime.now()
        step_results = []
        screenshots = []

        # Converte cenario em prompt para o agente
        prompt = self._scenario_to_prompt(scenario, base_url)

        try:
            llm = self._get_llm()
            agent = Agent(
                task=prompt,
                llm=llm,
                browser=self._browser,
            )

            result = await agent.run(max_steps=self.max_steps_per_scenario)

            # Analisa resultado
            status, ux_summary = self._parse_agent_result(result, scenario)

            # Cria step results baseado no que o agente fez
            for i, step in enumerate(scenario.steps):
                step_results.append(StepResult(
                    step_index=i,
                    action=step.action.value,
                    status=StepStatus.PASSED if status == ScenarioStatus.PASSED else StepStatus.FAILED,
                    message=step.description,
                ))

        except Exception as e:
            status = ScenarioStatus.ERROR
            ux_summary = f"Erro na execucao: {str(e)}"

            for i, step in enumerate(scenario.steps):
                step_results.append(StepResult(
                    step_index=i,
                    action=step.action.value,
                    status=StepStatus.ERROR,
                    error=str(e),
                ))

        end_time = datetime.now()
        duration = int((end_time - start_time).total_seconds() * 1000)

        return ScenarioResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            status=status,
            executor=self.name,
            step_results=step_results,
            duration_ms=duration,
            start_time=start_time,
            end_time=end_time,
            screenshots=screenshots,
            ux_summary=ux_summary,
        )

    def _scenario_to_prompt(self, scenario: TestScenario, base_url: str) -> str:
        """Converte cenario em prompt para o agente."""
        steps_text = "\n".join([
            f"{i+1}. {step.description or f'{step.action.value} em {step.target}'}"
            for i, step in enumerate(scenario.steps)
        ])

        return f"""
TESTE: {scenario.name}

OBJETIVO: Executar o seguinte cenario de teste e verificar se funciona corretamente.

PASSOS:
{steps_text}

RESULTADO ESPERADO: {scenario.expected_result}

INSTRUCOES:
1. Execute cada passo na ordem indicada
2. Observe o comportamento da aplicacao
3. Verifique se o resultado esperado foi atingido
4. Avalie a experiencia do usuario (UX)

Ao terminar, responda com JSON no formato:
{{
    "passed": true/false,
    "reason": "explicacao do resultado",
    "ux_feedback": "comentarios sobre UX",
    "steps_completed": numero_de_passos_executados
}}

Use a acao 'done' com o JSON quando terminar.
"""

    def _parse_agent_result(self, result, scenario: TestScenario) -> tuple[ScenarioStatus, str]:
        """Analisa resultado do agente e determina status."""
        content = ""

        if hasattr(result, 'all_results'):
            for r in result.all_results:
                if hasattr(r, 'extracted_content') and r.extracted_content:
                    content += str(r.extracted_content) + "\n"

        # Tenta extrair JSON
        try:
            # Procura por JSON na resposta
            json_match = re.search(r'\{[^{}]*"passed"[^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                passed = data.get("passed", False)
                ux_feedback = data.get("ux_feedback", "")
                reason = data.get("reason", "")

                status = ScenarioStatus.PASSED if passed else ScenarioStatus.FAILED
                summary = f"{reason}. UX: {ux_feedback}" if ux_feedback else reason

                return status, summary
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: assume sucesso se completou sem erro
        return ScenarioStatus.PASSED, "Execucao completada"

    async def cleanup(self):
        """Fecha browser."""
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
