"""
UI Analyzer - Agente de Teste Automatizado com IA

Agente completo de QA que testa aplicações web como um usuário real,
usando LLMs com visão para:
- Testes funcionais (navegação, cliques, formulários)
- Detecção de bugs (erros de console, elementos quebrados)
- Testes de performance (tempo de carregamento, responsividade)
- Análise de UX e acessibilidade
- Sugestões de melhoria automatizadas

Uso:
    from dumont_core.testing import UIAnalyzer

    analyzer = UIAnalyzer(base_url="http://localhost:8080")
    report = await analyzer.run_full_test()  # Teste completo

    # Ou testes específicos:
    report = await analyzer.test_functionality()
    report = await analyzer.test_performance()
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from browser_use import Agent, Browser, Controller
    from browser_use.browser.context import BrowserContext
    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False

try:
    from browser_use import ChatOpenAI as BrowserUseChatOpenAI
    BROWSER_USE_OPENAI_AVAILABLE = True
except ImportError:
    BROWSER_USE_OPENAI_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI as LangChainChatOpenAI
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_GOOGLE_AVAILABLE = True
except ImportError:
    LANGCHAIN_GOOGLE_AVAILABLE = False


class TestSeverity(Enum):
    """Severidade dos problemas encontrados."""
    CRITICAL = "critical"  # Bloqueia uso da aplicação
    HIGH = "high"          # Funcionalidade importante quebrada
    MEDIUM = "medium"      # Problema afeta UX mas não bloqueia
    LOW = "low"            # Sugestão de melhoria
    INFO = "info"          # Informação/observação


class TestCategory(Enum):
    """Categorias de testes."""
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    ACCESSIBILITY = "accessibility"
    UX = "ux"
    SECURITY = "security"
    VISUAL = "visual"


@dataclass
class TestIssue:
    """Representa um problema encontrado durante os testes."""
    title: str
    description: str
    severity: TestSeverity
    category: TestCategory
    location: str = ""
    screenshot_path: str = ""
    suggestion: str = ""
    raw_error: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "category": self.category.value,
            "location": self.location,
            "screenshot_path": self.screenshot_path,
            "suggestion": self.suggestion,
            "raw_error": self.raw_error
        }


@dataclass
class TestResult:
    """Resultado de um teste individual."""
    test_name: str
    passed: bool
    duration_ms: float
    issues: list[TestIssue] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    agent_output: str = ""

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "issues": [i.to_dict() for i in self.issues],
            "details": self.details,
            "agent_output": self.agent_output
        }


@dataclass
class TestReport:
    """Relatório completo de testes."""
    timestamp: str
    base_url: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    total_issues: int = 0
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    total_duration_ms: float = 0
    results: list[TestResult] = field(default_factory=list)
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "base_url": self.base_url,
            "summary": {
                "total_tests": self.total_tests,
                "passed": self.passed_tests,
                "failed": self.failed_tests,
                "total_issues": self.total_issues,
                "by_severity": {
                    "critical": self.critical_issues,
                    "high": self.high_issues,
                    "medium": self.medium_issues,
                    "low": self.low_issues
                },
                "total_duration_ms": self.total_duration_ms
            },
            "results": [r.to_dict() for r in self.results],
            "ai_summary": self.summary,
            "recommendations": self.recommendations
        }


class UIAnalyzer:
    """
    Agente de Teste Automatizado com IA.

    Executa testes completos em aplicações web usando Browser-Use e LLMs
    com capacidade de visão para análise inteligente.

    Attributes:
        base_url: URL base da aplicação a ser testada
        output_dir: Diretório para salvar relatórios e screenshots
        llm_model: Modelo de LLM a ser usado
        llm_provider: Provider do LLM (openrouter, openai, google)
        headless: Se deve rodar o browser em modo headless
        auth_config: Configuração de autenticação (login, senha, etc)
    """

    # Testes funcionais padrão
    DEFAULT_FUNCTIONAL_TESTS = [
        "navigation",
        "forms",
        "buttons",
        "links",
        "modals",
        "error_handling"
    ]

    def __init__(
        self,
        base_url: str = "http://localhost:80",
        output_dir: str = "/tmp/ui_tests",
        llm_model: str = "google/gemini-2.5-flash",
        llm_provider: str = "openrouter",
        headless: bool = True,
        auth_config: Optional[dict] = None,
        custom_tests: Optional[dict[str, str]] = None,
        timeout_ms: int = 30000
    ):
        """
        Inicializa o agente de testes.

        Args:
            base_url: URL base da aplicação
            output_dir: Diretório para relatórios
            llm_model: Modelo de LLM
            llm_provider: Provider (openrouter, openai, google)
            headless: Modo headless
            auth_config: Dict com credenciais {"username": "", "password": "", "login_url": ""}
            custom_tests: Dict com testes customizados {nome: task_description}
            timeout_ms: Timeout para operações em ms
        """
        if not BROWSER_USE_AVAILABLE:
            raise ImportError(
                "browser-use não está instalado. "
                "Instale com: pip install browser-use"
            )

        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.headless = headless
        self.auth_config = auth_config
        self.custom_tests = custom_tests or {}
        self.timeout_ms = timeout_ms
        self.browser: Optional[Browser] = None
        self.console_errors: list[str] = []
        self.network_errors: list[str] = []
        self.performance_metrics: dict = {}

    def _create_llm(self):
        """Cria a instância do LLM baseado no provider configurado."""
        if self.llm_provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY não configurada")

            if BROWSER_USE_OPENAI_AVAILABLE:
                return BrowserUseChatOpenAI(
                    model=self.llm_model,
                    temperature=0,
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
            elif LANGCHAIN_OPENAI_AVAILABLE:
                return LangChainChatOpenAI(
                    model=self.llm_model,
                    temperature=0,
                    openai_api_key=api_key,
                    openai_api_base="https://openrouter.ai/api/v1"
                )
            else:
                raise ImportError("Nenhuma biblioteca de LLM disponível")

        elif self.llm_provider == "google":
            if not LANGCHAIN_GOOGLE_AVAILABLE:
                raise ImportError("langchain-google-genai não instalado")
            return ChatGoogleGenerativeAI(
                model=self.llm_model,
                temperature=0
            )

        elif self.llm_provider == "openai":
            if BROWSER_USE_OPENAI_AVAILABLE:
                return BrowserUseChatOpenAI(model=self.llm_model, temperature=0)
            elif LANGCHAIN_OPENAI_AVAILABLE:
                return LangChainChatOpenAI(model=self.llm_model, temperature=0)
            else:
                raise ImportError("Nenhuma biblioteca de LLM disponível")

        else:
            raise ValueError(f"Provider não suportado: {self.llm_provider}")

    async def _setup_browser(self) -> Browser:
        """Configura o browser com captura de erros."""
        self.browser = Browser(
            headless=self.headless,
            disable_security=True
        )
        return self.browser

    async def _authenticate(self, agent: Agent) -> bool:
        """Executa autenticação se configurada."""
        if not self.auth_config:
            return True

        login_url = self.auth_config.get("login_url", f"{self.base_url}/login")
        username = self.auth_config.get("username", "")
        password = self.auth_config.get("password", "")

        auth_task = f"""
        Vá para {login_url} e faça login:
        1. Encontre o campo de usuário/email e digite: {username}
        2. Encontre o campo de senha e digite: {password}
        3. Clique no botão de login/entrar
        4. Aguarde o redirecionamento
        5. Confirme se o login foi bem sucedido (procure por elementos que indicam usuário logado)

        Retorne "LOGIN_SUCCESS" se conseguiu logar, ou "LOGIN_FAILED" + motivo se falhou.
        """

        result = await agent.run(max_steps=10)
        return "LOGIN_SUCCESS" in str(result)

    async def run_full_test(
        self,
        include_performance: bool = True,
        include_accessibility: bool = True,
        include_visual: bool = True,
        on_test_complete: Optional[Callable[[TestResult], None]] = None
    ) -> TestReport:
        """
        Executa suíte completa de testes.

        Args:
            include_performance: Incluir testes de performance
            include_accessibility: Incluir testes de acessibilidade
            include_visual: Incluir análise visual
            on_test_complete: Callback após cada teste

        Returns:
            TestReport com todos os resultados
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = TestReport(
            timestamp=timestamp,
            base_url=self.base_url
        )

        print(f"\n{'='*60}")
        print(f"INICIANDO TESTES - {self.base_url}")
        print(f"{'='*60}\n")

        browser = await self._setup_browser()
        llm = self._create_llm()

        try:
            # 1. Teste de carregamento inicial
            result = await self._test_initial_load(browser, llm, timestamp)
            report.results.append(result)
            if on_test_complete:
                on_test_complete(result)

            # 2. Testes funcionais
            functional_result = await self._test_functionality(browser, llm, timestamp)
            report.results.append(functional_result)
            if on_test_complete:
                on_test_complete(functional_result)

            # 3. Testes de performance
            if include_performance:
                perf_result = await self._test_performance(browser, llm, timestamp)
                report.results.append(perf_result)
                if on_test_complete:
                    on_test_complete(perf_result)

            # 4. Testes de acessibilidade
            if include_accessibility:
                a11y_result = await self._test_accessibility(browser, llm, timestamp)
                report.results.append(a11y_result)
                if on_test_complete:
                    on_test_complete(a11y_result)

            # 5. Análise visual/UX
            if include_visual:
                visual_result = await self._test_visual_ux(browser, llm, timestamp)
                report.results.append(visual_result)
                if on_test_complete:
                    on_test_complete(visual_result)

            # 6. Testes customizados
            for test_name, test_task in self.custom_tests.items():
                custom_result = await self._run_custom_test(
                    browser, llm, timestamp, test_name, test_task
                )
                report.results.append(custom_result)
                if on_test_complete:
                    on_test_complete(custom_result)

            # Calcular estatísticas
            self._calculate_report_stats(report)

            # Gerar sumário com IA
            report.summary = await self._generate_ai_summary(llm, report)
            report.recommendations = await self._generate_recommendations(llm, report)

        finally:
            await browser.close()

        # Salvar relatório
        report_path = self.output_dir / f"{timestamp}_full_report.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"TESTES CONCLUÍDOS")
        print(f"{'='*60}")
        print(f"Total: {report.total_tests} | Passou: {report.passed_tests} | Falhou: {report.failed_tests}")
        print(f"Issues: {report.total_issues} (Critical: {report.critical_issues}, High: {report.high_issues})")
        print(f"Relatório salvo em: {report_path}")

        return report

    async def _test_initial_load(
        self,
        browser: Browser,
        llm,
        timestamp: str
    ) -> TestResult:
        """Testa carregamento inicial da página."""
        print("\n[TEST] Carregamento Inicial...")
        start_time = time.time()
        issues = []

        task = f"""
        Vá para {self.base_url} e analise o carregamento inicial:

        1. MEÇA o tempo até a página estar completamente carregada
        2. VERIFIQUE se há erros visíveis (mensagens de erro, páginas 404/500)
        3. VERIFIQUE se todos os elementos principais carregaram (header, menu, conteúdo)
        4. CAPTURE qualquer erro no console do navegador
        5. TIRE um screenshot da página carregada

        IMPORTANTE: Relate problemas encontrados no formato:
        - [CRITICAL] Descrição do problema crítico
        - [HIGH] Descrição do problema grave
        - [MEDIUM] Descrição do problema médio
        - [LOW] Sugestão de melhoria

        Se tudo estiver OK, diga "LOAD_SUCCESS" e descreva o que viu.
        """

        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            save_conversation_path=str(
                self.output_dir / f"{timestamp}_initial_load.json"
            )
        )

        try:
            result = await agent.run(max_steps=10)
            duration = (time.time() - start_time) * 1000

            # Parse issues from result
            issues = self._parse_issues_from_result(str(result), TestCategory.FUNCTIONAL)

            passed = "LOAD_SUCCESS" in str(result) or len([
                i for i in issues if i.severity in [TestSeverity.CRITICAL, TestSeverity.HIGH]
            ]) == 0

            return TestResult(
                test_name="initial_load",
                passed=passed,
                duration_ms=duration,
                issues=issues,
                agent_output=str(result)
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            issues.append(TestIssue(
                title="Erro no teste de carregamento",
                description=str(e),
                severity=TestSeverity.CRITICAL,
                category=TestCategory.FUNCTIONAL,
                raw_error=str(e)
            ))
            return TestResult(
                test_name="initial_load",
                passed=False,
                duration_ms=duration,
                issues=issues
            )

    async def _test_functionality(
        self,
        browser: Browser,
        llm,
        timestamp: str
    ) -> TestResult:
        """Testa funcionalidades principais."""
        print("\n[TEST] Funcionalidades...")
        start_time = time.time()
        issues = []

        task = f"""
        Na página {self.base_url}, teste as funcionalidades principais:

        1. NAVEGAÇÃO:
           - Clique em todos os links de navegação principais
           - Verifique se levam para as páginas corretas
           - Teste o botão voltar do navegador

        2. BOTÕES:
           - Identifique todos os botões da interface
           - Clique nos botões e verifique se respondem
           - Verifique feedback visual (hover, click states)

        3. FORMULÁRIOS (se existirem):
           - Encontre campos de input
           - Tente preencher com dados válidos
           - Tente submeter e verifique resposta
           - Teste validações (campos obrigatórios, formatos)

        4. INTERAÇÕES:
           - Teste dropdowns, modais, tooltips
           - Verifique se fecham corretamente
           - Teste atalhos de teclado se houver

        5. ERROS:
           - Capture qualquer erro de JavaScript no console
           - Identifique elementos que não respondem
           - Note comportamentos inesperados

        REPORTE problemas no formato:
        - [CRITICAL] Funcionalidade quebrada que bloqueia uso
        - [HIGH] Bug importante
        - [MEDIUM] Problema de UX
        - [LOW] Sugestão

        Se tudo funcionar bem, diga "FUNCTIONAL_PASS".
        """

        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            save_conversation_path=str(
                self.output_dir / f"{timestamp}_functionality.json"
            )
        )

        try:
            result = await agent.run(max_steps=25)
            duration = (time.time() - start_time) * 1000

            issues = self._parse_issues_from_result(str(result), TestCategory.FUNCTIONAL)

            passed = "FUNCTIONAL_PASS" in str(result) or len([
                i for i in issues if i.severity in [TestSeverity.CRITICAL, TestSeverity.HIGH]
            ]) == 0

            return TestResult(
                test_name="functionality",
                passed=passed,
                duration_ms=duration,
                issues=issues,
                agent_output=str(result)
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            issues.append(TestIssue(
                title="Erro no teste funcional",
                description=str(e),
                severity=TestSeverity.CRITICAL,
                category=TestCategory.FUNCTIONAL,
                raw_error=str(e)
            ))
            return TestResult(
                test_name="functionality",
                passed=False,
                duration_ms=duration,
                issues=issues
            )

    async def _test_performance(
        self,
        browser: Browser,
        llm,
        timestamp: str
    ) -> TestResult:
        """Testa performance da aplicação."""
        print("\n[TEST] Performance...")
        start_time = time.time()
        issues = []

        task = f"""
        Analise a performance da aplicação em {self.base_url}:

        1. TEMPO DE CARREGAMENTO:
           - Recarregue a página e meça o tempo até estar interativa
           - Observe se há delays perceptíveis
           - Note elementos que demoram a carregar

        2. RESPONSIVIDADE:
           - Clique em elementos e meça tempo de resposta
           - Identifique operações que parecem lentas
           - Note se há travamentos ou lags

        3. ANIMAÇÕES:
           - Verifique se animações são suaves (60fps)
           - Identifique animações que "engasgam"

        4. RECURSOS:
           - Observe se há muitas requisições de rede
           - Note se imagens são muito pesadas
           - Identifique scripts que bloqueiam renderização

        MÉTRICAS a reportar:
        - Tempo de First Contentful Paint (estimado)
        - Tempo até página interativa
        - Elementos que demoram mais de 1s para responder

        REPORTE problemas:
        - [HIGH] Performance que afeta usabilidade (>3s de delay)
        - [MEDIUM] Delays perceptíveis (1-3s)
        - [LOW] Otimizações sugeridas

        Se performance for boa, diga "PERFORMANCE_PASS".
        """

        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            save_conversation_path=str(
                self.output_dir / f"{timestamp}_performance.json"
            )
        )

        try:
            result = await agent.run(max_steps=15)
            duration = (time.time() - start_time) * 1000

            issues = self._parse_issues_from_result(str(result), TestCategory.PERFORMANCE)

            passed = "PERFORMANCE_PASS" in str(result) or len([
                i for i in issues if i.severity == TestSeverity.HIGH
            ]) == 0

            return TestResult(
                test_name="performance",
                passed=passed,
                duration_ms=duration,
                issues=issues,
                agent_output=str(result)
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return TestResult(
                test_name="performance",
                passed=False,
                duration_ms=duration,
                issues=[TestIssue(
                    title="Erro no teste de performance",
                    description=str(e),
                    severity=TestSeverity.MEDIUM,
                    category=TestCategory.PERFORMANCE
                )]
            )

    async def _test_accessibility(
        self,
        browser: Browser,
        llm,
        timestamp: str
    ) -> TestResult:
        """Testa acessibilidade."""
        print("\n[TEST] Acessibilidade...")
        start_time = time.time()
        issues = []

        task = f"""
        Analise a acessibilidade da aplicação em {self.base_url}:

        1. CONTRASTE:
           - Verifique se textos têm contraste suficiente com o fundo
           - Identifique textos difíceis de ler

        2. TAMANHOS:
           - Fontes são legíveis? (mínimo 16px recomendado)
           - Botões/links são grandes o suficiente para clicar? (mínimo 44x44px)

        3. NAVEGAÇÃO POR TECLADO:
           - Pressione Tab para navegar pela página
           - Verifique se todos os elementos interativos são acessíveis
           - Verifique se há indicador visual de foco

        4. SEMÂNTICA:
           - Há textos alternativos em imagens?
           - Labels estão associados aos inputs?
           - Hierarquia de headings faz sentido?

        5. CORES:
           - Informações são transmitidas apenas por cor?
           - Há modo escuro/claro disponível?

        REPORTE problemas (WCAG 2.1):
        - [HIGH] Violação de nível A (crítico para acessibilidade)
        - [MEDIUM] Violação de nível AA
        - [LOW] Sugestões de melhoria

        Se acessibilidade for adequada, diga "A11Y_PASS".
        """

        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            save_conversation_path=str(
                self.output_dir / f"{timestamp}_accessibility.json"
            )
        )

        try:
            result = await agent.run(max_steps=15)
            duration = (time.time() - start_time) * 1000

            issues = self._parse_issues_from_result(str(result), TestCategory.ACCESSIBILITY)

            passed = "A11Y_PASS" in str(result) or len([
                i for i in issues if i.severity == TestSeverity.HIGH
            ]) == 0

            return TestResult(
                test_name="accessibility",
                passed=passed,
                duration_ms=duration,
                issues=issues,
                agent_output=str(result)
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return TestResult(
                test_name="accessibility",
                passed=False,
                duration_ms=duration,
                issues=[TestIssue(
                    title="Erro no teste de acessibilidade",
                    description=str(e),
                    severity=TestSeverity.MEDIUM,
                    category=TestCategory.ACCESSIBILITY
                )]
            )

    async def _test_visual_ux(
        self,
        browser: Browser,
        llm,
        timestamp: str
    ) -> TestResult:
        """Analisa aspectos visuais e UX."""
        print("\n[TEST] Visual/UX...")
        start_time = time.time()
        issues = []

        task = f"""
        Analise visualmente a interface em {self.base_url}:

        1. LAYOUT:
           - A hierarquia visual está clara?
           - O espaçamento é consistente?
           - Há alinhamento adequado dos elementos?

        2. CONSISTÊNCIA:
           - Cores seguem um padrão?
           - Fontes são consistentes?
           - Botões têm estilo uniforme?

        3. CLAREZA:
           - Fica claro o que cada elemento faz?
           - Há feedback visual para ações?
           - Estados de erro são evidentes?

        4. RESPONSIVIDADE:
           - Redimensione a janela para tamanho mobile (375px)
           - A interface se adapta?
           - Conteúdo fica cortado ou quebrado?

        5. USABILIDADE:
           - É fácil encontrar as funções principais?
           - A navegação é intuitiva?
           - Há call-to-actions claros?

        TIRE screenshots em diferentes estados e tamanhos.

        REPORTE problemas:
        - [HIGH] Problema grave de usabilidade
        - [MEDIUM] UX pode ser melhorada
        - [LOW] Sugestões estéticas

        Se visual/UX for bom, diga "UX_PASS".
        """

        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            save_conversation_path=str(
                self.output_dir / f"{timestamp}_visual_ux.json"
            )
        )

        try:
            result = await agent.run(max_steps=20)
            duration = (time.time() - start_time) * 1000

            issues = self._parse_issues_from_result(str(result), TestCategory.UX)

            passed = "UX_PASS" in str(result) or len([
                i for i in issues if i.severity == TestSeverity.HIGH
            ]) == 0

            return TestResult(
                test_name="visual_ux",
                passed=passed,
                duration_ms=duration,
                issues=issues,
                agent_output=str(result)
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return TestResult(
                test_name="visual_ux",
                passed=False,
                duration_ms=duration,
                issues=[TestIssue(
                    title="Erro na análise visual",
                    description=str(e),
                    severity=TestSeverity.LOW,
                    category=TestCategory.UX
                )]
            )

    async def _run_custom_test(
        self,
        browser: Browser,
        llm,
        timestamp: str,
        test_name: str,
        test_task: str
    ) -> TestResult:
        """Executa um teste customizado."""
        print(f"\n[TEST] Custom: {test_name}...")
        start_time = time.time()

        # Adiciona instruções de report ao task
        full_task = f"""
        {test_task}

        REPORTE problemas no formato:
        - [CRITICAL] Problema crítico
        - [HIGH] Problema grave
        - [MEDIUM] Problema médio
        - [LOW] Sugestão

        Se tudo estiver OK, diga "CUSTOM_PASS".
        """

        agent = Agent(
            task=full_task,
            llm=llm,
            browser=browser,
            save_conversation_path=str(
                self.output_dir / f"{timestamp}_{test_name}.json"
            )
        )

        try:
            result = await agent.run(max_steps=20)
            duration = (time.time() - start_time) * 1000

            issues = self._parse_issues_from_result(str(result), TestCategory.FUNCTIONAL)

            passed = "CUSTOM_PASS" in str(result) or len([
                i for i in issues if i.severity in [TestSeverity.CRITICAL, TestSeverity.HIGH]
            ]) == 0

            return TestResult(
                test_name=f"custom_{test_name}",
                passed=passed,
                duration_ms=duration,
                issues=issues,
                agent_output=str(result)
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return TestResult(
                test_name=f"custom_{test_name}",
                passed=False,
                duration_ms=duration,
                issues=[TestIssue(
                    title=f"Erro no teste {test_name}",
                    description=str(e),
                    severity=TestSeverity.MEDIUM,
                    category=TestCategory.FUNCTIONAL
                )]
            )

    def _parse_issues_from_result(
        self,
        result: str,
        default_category: TestCategory
    ) -> list[TestIssue]:
        """Extrai issues do resultado do agente."""
        issues = []

        severity_map = {
            "[CRITICAL]": TestSeverity.CRITICAL,
            "[HIGH]": TestSeverity.HIGH,
            "[MEDIUM]": TestSeverity.MEDIUM,
            "[LOW]": TestSeverity.LOW,
            "[INFO]": TestSeverity.INFO
        }

        for line in result.split("\n"):
            line = line.strip()
            if not line:
                continue

            for tag, severity in severity_map.items():
                if tag in line.upper():
                    description = line.replace(tag, "").strip()
                    # Remove variações do tag
                    for t in severity_map.keys():
                        description = description.replace(t, "").replace(t.lower(), "")
                    description = description.strip("- ")

                    if description:
                        issues.append(TestIssue(
                            title=description[:100],
                            description=description,
                            severity=severity,
                            category=default_category
                        ))
                    break

        return issues

    def _calculate_report_stats(self, report: TestReport):
        """Calcula estatísticas do relatório."""
        report.total_tests = len(report.results)
        report.passed_tests = sum(1 for r in report.results if r.passed)
        report.failed_tests = report.total_tests - report.passed_tests
        report.total_duration_ms = sum(r.duration_ms for r in report.results)

        all_issues = []
        for r in report.results:
            all_issues.extend(r.issues)

        report.total_issues = len(all_issues)
        report.critical_issues = sum(1 for i in all_issues if i.severity == TestSeverity.CRITICAL)
        report.high_issues = sum(1 for i in all_issues if i.severity == TestSeverity.HIGH)
        report.medium_issues = sum(1 for i in all_issues if i.severity == TestSeverity.MEDIUM)
        report.low_issues = sum(1 for i in all_issues if i.severity == TestSeverity.LOW)

    async def _generate_ai_summary(self, llm, report: TestReport) -> str:
        """Gera sumário com IA."""
        issues_text = ""
        for r in report.results:
            if r.issues:
                issues_text += f"\n{r.test_name}:\n"
                for i in r.issues:
                    issues_text += f"  - [{i.severity.value}] {i.description}\n"

        prompt = f"""
        Baseado nos resultados dos testes da aplicação {self.base_url}:

        Total de testes: {report.total_tests}
        Passou: {report.passed_tests}
        Falhou: {report.failed_tests}

        Issues encontradas:
        {issues_text if issues_text else "Nenhuma issue encontrada"}

        Gere um sumário executivo em português de 2-3 parágrafos sobre:
        1. Estado geral da aplicação
        2. Principais problemas encontrados
        3. Nível de qualidade/maturidade
        """

        try:
            response = await llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception:
            return "Sumário não disponível"

    async def _generate_recommendations(self, llm, report: TestReport) -> list[str]:
        """Gera recomendações de melhoria."""
        issues_text = ""
        for r in report.results:
            for i in r.issues:
                issues_text += f"- [{i.severity.value}] {i.category.value}: {i.description}\n"

        if not issues_text:
            return ["Aplicação está em bom estado. Continue monitorando."]

        prompt = f"""
        Baseado nas issues encontradas nos testes:

        {issues_text}

        Gere uma lista de 5-10 recomendações práticas de melhoria,
        ordenadas por prioridade (mais críticas primeiro).

        Formato: Uma recomendação por linha, começando com "- ".
        """

        try:
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return [
                line.strip("- ").strip()
                for line in content.split("\n")
                if line.strip().startswith("-")
            ]
        except Exception:
            return ["Revise as issues reportadas e corrija por ordem de severidade"]

    # ================================================================
    # Métodos de conveniência para testes individuais
    # ================================================================

    async def test_functionality(self) -> TestResult:
        """Executa apenas testes funcionais."""
        browser = await self._setup_browser()
        llm = self._create_llm()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            return await self._test_functionality(browser, llm, timestamp)
        finally:
            await browser.close()

    async def test_performance(self) -> TestResult:
        """Executa apenas testes de performance."""
        browser = await self._setup_browser()
        llm = self._create_llm()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            return await self._test_performance(browser, llm, timestamp)
        finally:
            await browser.close()

    async def test_accessibility(self) -> TestResult:
        """Executa apenas testes de acessibilidade."""
        browser = await self._setup_browser()
        llm = self._create_llm()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            return await self._test_accessibility(browser, llm, timestamp)
        finally:
            await browser.close()

    async def test_visual(self) -> TestResult:
        """Executa apenas análise visual/UX."""
        browser = await self._setup_browser()
        llm = self._create_llm()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            return await self._test_visual_ux(browser, llm, timestamp)
        finally:
            await browser.close()

    async def run_custom_test(self, name: str, task: str) -> TestResult:
        """Executa um teste customizado."""
        browser = await self._setup_browser()
        llm = self._create_llm()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            return await self._run_custom_test(browser, llm, timestamp, name, task)
        finally:
            await browser.close()

    def add_test(self, name: str, task: str):
        """Adiciona um teste customizado à suíte."""
        self.custom_tests[name] = task

    # ================================================================
    # Compatibilidade com API antiga (analyze)
    # ================================================================

    async def analyze(
        self,
        aspects: Optional[list[str]] = None,
        max_steps_per_aspect: int = 15,
        on_aspect_complete: Optional[Callable[[str, Any], None]] = None
    ) -> dict:
        """
        Método de compatibilidade - executa análise de aspectos visuais.
        Para testes completos, use run_full_test().
        """
        report = await self.run_full_test(
            include_performance=True,
            include_accessibility=True,
            include_visual=True
        )
        return report.to_dict()


async def main():
    """Exemplo de uso do UIAnalyzer."""

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERRO: OPENROUTER_API_KEY não configurada")
        return

    # Criar analisador
    analyzer = UIAnalyzer(
        base_url="http://localhost:80",
        output_dir="/tmp/ui_tests",
        llm_model="google/gemini-2.5-flash-preview-05-20",
        llm_provider="openrouter",
        headless=False  # Modo visual para debug
    )

    # Adicionar teste customizado específico da aplicação
    analyzer.add_test(
        "chat_interaction",
        """
        Teste a funcionalidade de chat:
        1. Encontre o campo de input de mensagens
        2. Digite "Olá, teste de chat"
        3. Envie a mensagem (Enter ou botão)
        4. Verifique se a mensagem aparece no histórico
        5. Aguarde resposta do sistema
        6. Verifique se a resposta é exibida corretamente
        """
    )

    # Executar suíte completa
    report = await analyzer.run_full_test()

    # Mostrar sumário
    print("\n" + "="*60)
    print("SUMÁRIO")
    print("="*60)
    print(report.summary)

    print("\nRECOMENDAÇÕES:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"{i}. {rec}")


if __name__ == "__main__":
    asyncio.run(main())
