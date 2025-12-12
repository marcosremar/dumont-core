"""
Explorador baseado em Browser-Use + LLM.

Usa Browser-Use com LLM (como Gemini Flash) para explorar a aplicacao
como um usuario real faria, focando em fluxos de usuario, UX e edge cases.
"""

import os
import re
import json
import uuid
from datetime import datetime
from typing import Optional
from pathlib import Path

from .base_explorer import BaseExplorer
from ..models.discovery import (
    DiscoveryResult,
    Feature,
    UserFlow,
    ElementType,
    FeaturePriority,
)


# Prompt de exploracao para o LLM
EXPLORATION_PROMPT = """
Voce e um testador QA explorando uma aplicacao web para descobrir o que precisa ser testado.

OBJETIVO: Explore a aplicacao em {url} e documente:
1. Todas as features/funcionalidades visiveis (botoes, campos, links, etc)
2. Fluxos de usuario possiveis
3. Potenciais edge cases e problemas de UX
4. Questoes de acessibilidade

INSTRUCOES:
- Navegue pela aplicacao interagindo com diferentes elementos
- Observe como a interface responde
- Documente suas descobertas de forma estruturada
- Foque em comportamento observado, nao em codigo

Ao terminar sua exploracao, responda com um JSON no seguinte formato:
{{
    "features": [
        {{
            "name": "Nome da Feature",
            "element_type": "button|input|link|text|form|panel|other",
            "description": "O que esta feature faz",
            "ux_notes": "Observacoes sobre usabilidade",
            "priority": "critical|high|medium|low"
        }}
    ],
    "user_flows": [
        {{
            "name": "Nome do Fluxo",
            "steps": ["Passo 1", "Passo 2", ...],
            "observed_behavior": "O que acontece ao executar",
            "ux_notes": "Observacoes sobre a experiencia"
        }}
    ],
    "edge_cases": ["Caso 1", "Caso 2", ...],
    "accessibility_issues": ["Issue 1", "Issue 2", ...]
}}

Use a acao 'done' com o JSON quando terminar.
"""


class BrowserUseExplorer(BaseExplorer):
    """
    Explorador que usa Browser-Use com LLM para descoberta.

    Foca em:
    - Fluxos de usuario naturais
    - Insights de UX (o que confunde, o que funciona bem)
    - Edge cases comportamentais
    - Descricoes em linguagem natural
    """

    def __init__(
        self,
        base_url: str,
        output_dir: Optional[str] = None,
        headless: bool = True,
        timeout: int = 300,
        llm_model: str = "google/gemini-2.5-flash",
        llm_provider: str = "openrouter",
        max_steps: int = 30,
    ):
        super().__init__(base_url, output_dir, headless, timeout)
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.max_steps = max_steps
        self._agent = None
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
                temperature=0.3,
            )
        elif self.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.llm_model,
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                temperature=0.3,
            )
        elif self.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.llm_model,
                api_key=os.environ.get("OPENAI_API_KEY"),
                temperature=0.3,
            )
        else:
            raise ValueError(f"Provider desconhecido: {self.llm_provider}")

    async def explore(self) -> DiscoveryResult:
        """
        Executa exploracao usando Browser-Use.

        Returns:
            DiscoveryResult com features, fluxos e insights descobertos
        """
        from browser_use import Agent, Browser, BrowserConfig

        # Configura browser
        browser_config = BrowserConfig(headless=self.headless)
        self._browser = Browser(config=browser_config)

        # Configura output
        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Cria agente
        llm = self._get_llm()
        prompt = EXPLORATION_PROMPT.format(url=self.base_url)

        self._agent = Agent(
            task=prompt,
            llm=llm,
            browser=self._browser,
        )

        # Executa exploracao
        try:
            result = await self._agent.run(max_steps=self.max_steps)
            return self._parse_result(result)
        finally:
            await self.cleanup()

    def _parse_result(self, result) -> DiscoveryResult:
        """Converte resultado do agente em DiscoveryResult."""
        features = []
        user_flows = []
        edge_cases = []
        accessibility_issues = []
        screenshots = []

        # Tenta extrair JSON estruturado da resposta
        raw_content = ""
        if hasattr(result, 'all_results'):
            for r in result.all_results:
                if hasattr(r, 'extracted_content') and r.extracted_content:
                    raw_content += str(r.extracted_content) + "\n"

        # Procura por JSON na resposta
        json_data = self._extract_json(raw_content)

        if json_data:
            # Parse features
            for f in json_data.get("features", []):
                try:
                    element_type = self._map_element_type(f.get("element_type", "other"))
                    priority = self._map_priority(f.get("priority", "medium"))

                    features.append(Feature(
                        id=f"bu_{uuid.uuid4().hex[:8]}",
                        name=f.get("name", "Unknown"),
                        element_type=element_type,
                        priority=priority,
                        description=f.get("description"),
                        ux_notes=f.get("ux_notes"),
                        source="browseruse",
                    ))
                except Exception:
                    pass

            # Parse user flows
            for flow in json_data.get("user_flows", []):
                try:
                    user_flows.append(UserFlow(
                        name=flow.get("name", "Unknown Flow"),
                        steps=flow.get("steps", []),
                        observed_behavior=flow.get("observed_behavior", ""),
                        ux_notes=flow.get("ux_notes"),
                        source="browseruse",
                    ))
                except Exception:
                    pass

            edge_cases = json_data.get("edge_cases", [])
            accessibility_issues = json_data.get("accessibility_issues", [])

        return DiscoveryResult(
            source="browseruse",
            timestamp=datetime.now(),
            base_url=self.base_url,
            features=features,
            user_flows=user_flows,
            edge_cases=edge_cases,
            accessibility_issues=accessibility_issues,
            raw_data={"content": raw_content, "parsed": json_data},
            screenshots=screenshots,
        )

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extrai JSON de texto."""
        # Tenta encontrar JSON no texto
        patterns = [
            r'\{[\s\S]*"features"[\s\S]*\}',
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_str = match.group(1) if match.lastindex else match.group(0)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue

        return None

    def _map_element_type(self, type_str: str) -> ElementType:
        """Mapeia string para ElementType."""
        mapping = {
            "button": ElementType.BUTTON,
            "input": ElementType.INPUT,
            "link": ElementType.LINK,
            "text": ElementType.TEXT,
            "image": ElementType.IMAGE,
            "form": ElementType.FORM,
            "list": ElementType.LIST,
            "table": ElementType.TABLE,
            "modal": ElementType.MODAL,
            "menu": ElementType.MENU,
            "tab": ElementType.TAB,
            "panel": ElementType.PANEL,
        }
        return mapping.get(type_str.lower(), ElementType.OTHER)

    def _map_priority(self, priority_str: str) -> FeaturePriority:
        """Mapeia string para FeaturePriority."""
        mapping = {
            "critical": FeaturePriority.CRITICAL,
            "high": FeaturePriority.HIGH,
            "medium": FeaturePriority.MEDIUM,
            "low": FeaturePriority.LOW,
        }
        return mapping.get(priority_str.lower(), FeaturePriority.MEDIUM)

    async def cleanup(self):
        """Fecha browser e libera recursos."""
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        self._agent = None
