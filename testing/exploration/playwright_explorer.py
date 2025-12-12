"""
Explorador baseado em Playwright.

Usa Playwright para analise tecnica profunda da aplicacao,
focando em seletores, acessibilidade, DOM e APIs.
"""

import uuid
import re
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


class PlaywrightExplorer(BaseExplorer):
    """
    Explorador que usa Playwright para analise tecnica.

    Foca em:
    - Seletores CSS/XPath precisos
    - Analise de acessibilidade (WCAG)
    - Network requests e APIs
    - Estado do DOM
    """

    def __init__(
        self,
        base_url: str,
        output_dir: Optional[str] = None,
        headless: bool = True,
        timeout: int = 300,
    ):
        super().__init__(base_url, output_dir, headless, timeout)
        self._browser = None
        self._page = None

    @property
    def name(self) -> str:
        return "playwright"

    async def explore(self) -> DiscoveryResult:
        """
        Executa exploracao usando Playwright.

        Returns:
            DiscoveryResult com seletores, acessibilidade e APIs
        """
        from playwright.async_api import async_playwright

        features = []
        user_flows = []
        accessibility_issues = []
        api_endpoints = []
        screenshots = []

        # Configura output
        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        async with async_playwright() as p:
            # Lanca browser
            self._browser = await p.chromium.launch(headless=self.headless)
            context = await self._browser.new_context()
            self._page = await context.new_page()

            # Captura requests de API
            captured_apis = []
            self._page.on("request", lambda req: captured_apis.append({
                "url": req.url,
                "method": req.method,
                "resource_type": req.resource_type,
            }))

            try:
                # Navega para a pagina
                await self._page.goto(self.base_url, wait_until="networkidle", timeout=self.timeout * 1000)

                # Screenshot inicial
                if self.output_dir:
                    screenshot_path = str(Path(self.output_dir) / "playwright_initial.png")
                    await self._page.screenshot(path=screenshot_path, full_page=True)
                    screenshots.append(screenshot_path)

                # Analisa elementos interativos
                features = await self._analyze_interactive_elements()

                # Analisa acessibilidade
                accessibility_issues = await self._check_accessibility()

                # Processa APIs capturadas
                api_endpoints = self._process_apis(captured_apis)

                # Gera fluxos basicos a partir dos elementos
                user_flows = self._generate_basic_flows(features)

            finally:
                await self._browser.close()

        return DiscoveryResult(
            source="playwright",
            timestamp=datetime.now(),
            base_url=self.base_url,
            features=features,
            user_flows=user_flows,
            accessibility_issues=accessibility_issues,
            api_endpoints=api_endpoints,
            screenshots=screenshots,
        )

    async def _analyze_interactive_elements(self) -> list[Feature]:
        """Analisa elementos interativos na pagina."""
        features = []

        # Botoes
        buttons = await self._page.query_selector_all("button, [role='button'], input[type='submit'], input[type='button']")
        for btn in buttons:
            try:
                text = await btn.text_content() or ""
                aria_label = await btn.get_attribute("aria-label") or ""
                selector = await self._get_best_selector(btn)

                features.append(Feature(
                    id=f"pw_{uuid.uuid4().hex[:8]}",
                    name=text.strip() or aria_label or "Button",
                    element_type=ElementType.BUTTON,
                    selector=selector,
                    accessible_name=aria_label or text.strip(),
                    aria_role="button",
                    source="playwright",
                    priority=FeaturePriority.HIGH,
                ))
            except Exception:
                pass

        # Inputs
        inputs = await self._page.query_selector_all("input:not([type='hidden']):not([type='submit']):not([type='button']), textarea")
        for inp in inputs:
            try:
                placeholder = await inp.get_attribute("placeholder") or ""
                aria_label = await inp.get_attribute("aria-label") or ""
                input_type = await inp.get_attribute("type") or "text"
                selector = await self._get_best_selector(inp)
                name = await inp.get_attribute("name") or ""

                features.append(Feature(
                    id=f"pw_{uuid.uuid4().hex[:8]}",
                    name=placeholder or aria_label or name or f"Input ({input_type})",
                    element_type=ElementType.INPUT,
                    selector=selector,
                    accessible_name=aria_label or placeholder,
                    source="playwright",
                    priority=FeaturePriority.HIGH,
                ))
            except Exception:
                pass

        # Links
        links = await self._page.query_selector_all("a[href]")
        for link in links:
            try:
                text = await link.text_content() or ""
                href = await link.get_attribute("href") or ""

                # Ignora links vazios ou de ancora
                if not text.strip() or href.startswith("#"):
                    continue

                selector = await self._get_best_selector(link)

                features.append(Feature(
                    id=f"pw_{uuid.uuid4().hex[:8]}",
                    name=text.strip()[:50],
                    element_type=ElementType.LINK,
                    selector=selector,
                    source="playwright",
                    priority=FeaturePriority.MEDIUM,
                ))
            except Exception:
                pass

        return features

    async def _get_best_selector(self, element) -> str:
        """Gera o melhor seletor possivel para um elemento."""
        try:
            # Tenta ID primeiro
            element_id = await element.get_attribute("id")
            if element_id:
                return f"#{element_id}"

            # Tenta data-testid
            testid = await element.get_attribute("data-testid")
            if testid:
                return f"[data-testid='{testid}']"

            # Tenta aria-label
            aria_label = await element.get_attribute("aria-label")
            if aria_label:
                return f"[aria-label='{aria_label}']"

            # Tenta placeholder
            placeholder = await element.get_attribute("placeholder")
            if placeholder:
                return f"[placeholder='{placeholder}']"

            # Fallback para texto
            text = await element.text_content()
            if text and len(text.strip()) < 50:
                tag = await element.evaluate("el => el.tagName.toLowerCase()")
                return f"{tag}:has-text('{text.strip()}')"

            return "/* selector not found */"
        except Exception:
            return "/* error getting selector */"

    async def _check_accessibility(self) -> list[str]:
        """Verifica problemas de acessibilidade basicos."""
        issues = []

        try:
            # Imagens sem alt
            images_without_alt = await self._page.query_selector_all("img:not([alt])")
            if images_without_alt:
                issues.append(f"{len(images_without_alt)} imagem(ns) sem atributo alt")

            # Inputs sem label
            inputs = await self._page.query_selector_all("input:not([type='hidden'])")
            for inp in inputs:
                try:
                    inp_id = await inp.get_attribute("id")
                    aria_label = await inp.get_attribute("aria-label")
                    placeholder = await inp.get_attribute("placeholder")

                    if inp_id:
                        label = await self._page.query_selector(f"label[for='{inp_id}']")
                        if not label and not aria_label:
                            issues.append(f"Input sem label associado: {inp_id}")
                    elif not aria_label and not placeholder:
                        issues.append("Input sem identificacao acessivel")
                except Exception:
                    pass

            # Contraste (verificacao basica)
            # TODO: Implementar verificacao de contraste real

            # Links sem texto
            empty_links = await self._page.query_selector_all("a:not(:has-text)")
            for link in empty_links:
                aria_label = await link.get_attribute("aria-label")
                if not aria_label:
                    issues.append("Link sem texto ou aria-label")

            # Heading hierarchy
            headings = await self._page.query_selector_all("h1, h2, h3, h4, h5, h6")
            if headings:
                heading_levels = []
                for h in headings:
                    tag = await h.evaluate("el => el.tagName")
                    level = int(tag[1])
                    heading_levels.append(level)

                # Verifica se comeca com h1
                if heading_levels and heading_levels[0] != 1:
                    issues.append("Pagina nao comeca com h1")

                # Verifica pulos de nivel
                for i in range(1, len(heading_levels)):
                    if heading_levels[i] - heading_levels[i-1] > 1:
                        issues.append(f"Pulo de nivel de heading: h{heading_levels[i-1]} para h{heading_levels[i]}")
                        break

        except Exception as e:
            issues.append(f"Erro na analise de acessibilidade: {str(e)}")

        return issues

    def _process_apis(self, captured_apis: list[dict]) -> list[dict]:
        """Processa e filtra APIs capturadas."""
        api_endpoints = []
        seen_urls = set()

        for api in captured_apis:
            url = api["url"]

            # Filtra recursos estaticos
            if api["resource_type"] in ["image", "stylesheet", "font", "script"]:
                continue

            # Filtra URLs ja vistas
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Identifica tipo
            api_type = "rest"
            if "ws://" in url or "wss://" in url:
                api_type = "websocket"
            elif "/graphql" in url:
                api_type = "graphql"

            # Extrai path relativo
            path = url.replace(self.base_url, "")
            if path.startswith("http"):
                # URL externa
                continue

            api_endpoints.append({
                "path": path or "/",
                "method": api["method"],
                "type": api_type,
                "full_url": url,
            })

        return api_endpoints

    def _generate_basic_flows(self, features: list[Feature]) -> list[UserFlow]:
        """Gera fluxos basicos a partir dos elementos descobertos."""
        flows = []

        # Encontra inputs e botoes
        inputs = [f for f in features if f.element_type == ElementType.INPUT]
        buttons = [f for f in features if f.element_type == ElementType.BUTTON]

        # Se tem inputs e botoes, cria fluxo de formulario
        if inputs and buttons:
            flow = UserFlow(
                name="Preenchimento de Formulario",
                steps=[f"Preencher campo: {inp.name}" for inp in inputs[:3]] +
                      [f"Clicar botao: {buttons[0].name}"],
                observed_behavior="Formulario enviado",
                source="playwright",
                feature_ids=[f.id for f in inputs[:3] + buttons[:1]],
            )
            flows.append(flow)

        # Se tem links de navegacao, cria fluxo de navegacao
        links = [f for f in features if f.element_type == ElementType.LINK]
        if links:
            flow = UserFlow(
                name="Navegacao por Links",
                steps=[f"Clicar link: {link.name}" for link in links[:3]],
                observed_behavior="Navegacao entre paginas",
                source="playwright",
                feature_ids=[f.id for f in links[:3]],
            )
            flows.append(flow)

        return flows

    async def cleanup(self):
        """Fecha browser."""
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        self._page = None
