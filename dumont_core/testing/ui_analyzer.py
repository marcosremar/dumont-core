"""
UI Analyzer - Módulo de teste de UI usando Browser-Use

Analisa interfaces web como um usuário real faria, usando LLMs com visão
para identificar problemas de UX, acessibilidade e layout.

Pode ser usado por qualquer aplicação Dumont para testes automatizados de UI.

Uso:
    from dumont_core.testing import UIAnalyzer

    analyzer = UIAnalyzer(base_url="http://localhost:8080")
    report = await analyzer.analyze()  # Analisa todos os aspectos

    # Ou aspectos específicos:
    report = await analyzer.analyze(aspects=["layout_geral", "responsividade"])
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

try:
    # browser-use 0.10+ uses lazy imports
    from browser_use import Agent, Browser, BrowserProfile, ChatOpenAI as BrowserUseChatOpenAI
    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False

# langchain imports are optional - browser-use has its own ChatOpenAI
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


class UIAnalyzer:
    """
    Analisador de UI usando Browser-Use com visão de LLM.

    Navega pela aplicação como um usuário real e usa um LLM com capacidade
    de visão (GPT-4o, Claude, etc) para analisar aspectos visuais da interface.

    Attributes:
        base_url: URL base da aplicação a ser testada
        output_dir: Diretório para salvar relatórios e screenshots
        llm_model: Modelo de LLM a ser usado (padrão: google/gemini-2.5-flash via OpenRouter)
        llm_provider: Provider do LLM (openrouter, openai, google)
        headless: Se deve rodar o browser em modo headless
    """

    # Aspectos padrão de análise
    DEFAULT_ASPECTS = [
        "layout_geral",
        "estado_idle",
        "execucao_tarefa",
        "painel_chat",
        "configuracoes",
        "responsividade",
        "acessibilidade"
    ]

    def __init__(
        self,
        base_url: str = "http://localhost:80",
        output_dir: str = "/tmp/ui_analysis",
        llm_model: str = "google/gemini-2.5-flash",
        llm_provider: str = "openrouter",
        headless: bool = True,
        custom_tasks: Optional[dict[str, str]] = None
    ):
        """
        Inicializa o analisador de UI.

        Args:
            base_url: URL base da aplicação a ser testada
            output_dir: Diretório para salvar relatórios
            llm_model: Modelo de LLM (ex: google/gemini-2.5-flash)
            llm_provider: Provider do LLM (openrouter, openai, google)
            headless: Se True, roda browser sem interface gráfica
            custom_tasks: Dict opcional com tasks customizadas {aspect: task_description}
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
        self.custom_tasks = custom_tasks or {}
        self.results: list[dict] = []

    def _create_llm(self):
        """Cria a instância do LLM baseado no provider configurado."""
        if self.llm_provider == "openrouter":
            # OpenRouter usa a API compatível com OpenAI
            # Primeiro tenta usar browser-use ChatOpenAI, depois langchain
            if BROWSER_USE_AVAILABLE:
                return BrowserUseChatOpenAI(
                    model=self.llm_model,
                    temperature=0,
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1"
                )
            elif LANGCHAIN_OPENAI_AVAILABLE:
                return LangChainChatOpenAI(
                    model=self.llm_model,
                    temperature=0,
                    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
                    openai_api_base="https://openrouter.ai/api/v1"
                )
            else:
                raise ImportError(
                    "Nenhuma biblioteca de LLM disponível. "
                    "Instale browser-use ou langchain-openai"
                )
        elif self.llm_provider == "google":
            if not LANGCHAIN_GOOGLE_AVAILABLE:
                raise ImportError(
                    "langchain-google-genai não está instalado. "
                    "Instale com: pip install langchain-google-genai"
                )
            return ChatGoogleGenerativeAI(
                model=self.llm_model,
                temperature=0
            )
        elif self.llm_provider == "openai":
            if BROWSER_USE_AVAILABLE:
                return BrowserUseChatOpenAI(
                    model=self.llm_model,
                    temperature=0
                )
            elif LANGCHAIN_OPENAI_AVAILABLE:
                return LangChainChatOpenAI(
                    model=self.llm_model,
                    temperature=0
                )
            else:
                raise ImportError(
                    "Nenhuma biblioteca de LLM disponível. "
                    "Instale browser-use ou langchain-openai"
                )
        else:
            raise ValueError(f"Provider não suportado: {self.llm_provider}")

    async def analyze(
        self,
        aspects: Optional[list[str]] = None,
        max_steps_per_aspect: int = 15,
        on_aspect_complete: Optional[Callable[[str, Any], None]] = None
    ) -> dict:
        """
        Executa análise completa da UI.

        Args:
            aspects: Lista de aspectos para analisar. Se None, analisa todos os DEFAULT_ASPECTS.
            max_steps_per_aspect: Número máximo de steps do agente por aspecto
            on_aspect_complete: Callback chamado ao completar cada aspecto

        Returns:
            Dict com relatório completo da análise
        """
        if aspects is None:
            aspects = self.DEFAULT_ASPECTS

        # Configura o browser (browser-use 0.10+ API)
        browser = Browser(
            headless=self.headless,
            disable_security=True
        )

        # Configura o LLM
        llm = self._create_llm()
        print(f"Usando LLM: {self.llm_provider}/{self.llm_model}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []

        try:
            for aspect in aspects:
                print(f"\n{'='*60}")
                print(f"Analisando: {aspect}")
                print('='*60)

                task = self._get_analysis_task(aspect)

                agent = Agent(
                    task=task,
                    llm=llm,
                    browser=browser,
                    save_conversation_path=str(
                        self.output_dir / f"{timestamp}_{aspect}_conversation.json"
                    )
                )

                result = await agent.run(max_steps=max_steps_per_aspect)

                result_entry = {
                    "aspect": aspect,
                    "task": task,
                    "result": result,
                    "timestamp": timestamp
                }
                self.results.append(result_entry)

                print(f"\nResultado para {aspect}:")
                print(result)

                if on_aspect_complete:
                    on_aspect_complete(aspect, result)

        finally:
            await browser.stop()

        # Gera relatório final
        report = self._generate_report()
        report_path = self.output_dir / f"{timestamp}_ui_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n\nRelatório salvo em: {report_path}")
        return report

    async def analyze_single(self, aspect: str, max_steps: int = 15) -> dict:
        """
        Analisa um único aspecto da UI.

        Args:
            aspect: Nome do aspecto a analisar
            max_steps: Número máximo de steps

        Returns:
            Dict com resultado da análise
        """
        results = await self.analyze(aspects=[aspect], max_steps_per_aspect=max_steps)
        return results.get("results", [{}])[0] if results.get("results") else {}

    async def run_custom_task(self, task_description: str, max_steps: int = 20) -> Any:
        """
        Executa uma tarefa customizada de teste.

        Args:
            task_description: Descrição da tarefa para o agente executar
            max_steps: Número máximo de steps

        Returns:
            Resultado da execução do agente
        """
        # Configura o browser (browser-use 0.10+ API)
        browser = Browser(
            headless=self.headless,
            disable_security=True
        )

        llm = self._create_llm()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            agent = Agent(
                task=task_description,
                llm=llm,
                browser=browser,
                save_conversation_path=str(
                    self.output_dir / f"{timestamp}_custom_conversation.json"
                )
            )
            result = await agent.run(max_steps=max_steps)
            return result
        finally:
            await browser.stop()

    def add_custom_aspect(self, name: str, task: str) -> None:
        """
        Adiciona um aspecto customizado de análise.

        Args:
            name: Nome do aspecto
            task: Descrição da tarefa de análise
        """
        self.custom_tasks[name] = task

    def _get_analysis_task(self, aspect: str) -> str:
        """Retorna a task de análise para cada aspecto."""

        # Primeiro verifica tasks customizadas
        if aspect in self.custom_tasks:
            return self.custom_tasks[aspect]

        tasks = {
            "layout_geral": f"""
Vá para {self.base_url} e analise o layout geral da interface.

Avalie:
1. Proporção entre áreas principais da interface
2. Uso do espaço - há áreas desperdiçadas?
3. Hierarquia visual - o que chama mais atenção?
4. Consistência visual - cores, fontes, espaçamentos
5. Clareza dos elementos - botões, labels, ícones

Tire um screenshot e liste:
- 3 pontos positivos
- 3 pontos que podem melhorar
- Sugestões específicas de mudança
""",

            "estado_idle": f"""
Vá para {self.base_url} e analise o estado idle (sem tarefa em execução).

Avalie:
1. A mensagem de estado inativo está clara?
2. Os botões de ação rápida são visíveis e convidativos?
3. O campo de input está bem posicionado?
4. Há feedback claro de que o sistema está pronto?
5. A imagem/animação de idle é apropriada?

Tire um screenshot e liste:
- O que funciona bem no estado idle
- O que pode confundir o usuário
- Sugestões de melhoria
""",

            "execucao_tarefa": f"""
Vá para {self.base_url} e:
1. Digite uma tarefa simples no campo de input (ex: "Pesquise o preço do Bitcoin")
2. Envie a tarefa
3. Observe a execução

Analise durante a execução:
1. Os steps de progresso estão visíveis?
2. A barra de progresso é clara?
3. As mensagens são legíveis?
4. O screenshot/preview atualiza em tempo real?
5. O botão de parar está acessível?

Tire screenshots durante a execução e liste:
- Feedback visual durante execução
- Clareza da progressão
- Problemas de UX identificados
""",

            "painel_chat": f"""
Vá para {self.base_url} e analise o painel de chat/mensagens.

Avalie:
1. O histórico de mensagens é legível?
2. A distinção entre mensagens do usuário e do sistema é clara?
3. O scroll funciona bem com muitas mensagens?
4. O campo de input é confortável para digitar?
5. Os botões de ação são intuitivos?

Tire um screenshot e liste:
- Usabilidade do chat
- Problemas de legibilidade
- Sugestões de melhoria
""",

            "configuracoes": f"""
Vá para {self.base_url} e procure por configurações ou opções.

Analise:
1. As opções são claras e bem organizadas?
2. Os controles (toggles, selects) são intuitivos?
3. Há tooltips ou explicações para cada opção?
4. As mudanças são aplicadas imediatamente ou precisam de confirmação?

Tire um screenshot e liste:
- Clareza das configurações
- O que está faltando
- Sugestões de melhoria
""",

            "responsividade": f"""
Vá para {self.base_url} e teste a responsividade:

1. Primeiro veja em tela cheia (1920x1080)
2. Depois redimensione para tablet (768px de largura)
3. Por fim, para mobile (375px de largura)

Para cada tamanho, avalie:
- Os elementos se reorganizam corretamente?
- O conteúdo continua legível?
- Os botões continuam clicáveis?
- Há overflow ou elementos cortados?

Liste problemas de responsividade encontrados.
""",

            "acessibilidade": f"""
Vá para {self.base_url} e avalie a acessibilidade:

1. Contraste de cores - textos são legíveis?
2. Tamanho de fonte - é confortável para leitura?
3. Áreas clicáveis - são grandes o suficiente (mínimo 44x44px)?
4. Feedback visual - hover states, focus states?
5. Navegação por teclado - dá pra usar sem mouse?

Liste:
- Problemas de acessibilidade identificados
- Sugestões de melhoria para WCAG compliance
"""
        }

        return tasks.get(aspect, f"Vá para {self.base_url} e analise a interface.")

    def _generate_report(self) -> dict:
        """Gera relatório consolidado da análise."""
        return {
            "summary": {
                "total_aspects_analyzed": len(self.results),
                "timestamp": datetime.now().isoformat(),
                "base_url": self.base_url,
                "llm_model": self.llm_model
            },
            "results": self.results,
            "recommendations": self._extract_recommendations()
        }

    def _extract_recommendations(self) -> list:
        """Extrai recomendações de todos os resultados."""
        recommendations = []
        for r in self.results:
            if r.get("result"):
                recommendations.append({
                    "aspect": r["aspect"],
                    "findings": str(r["result"])[:500]  # Primeiros 500 chars
                })
        return recommendations


async def main():
    """Exemplo de uso do UIAnalyzer."""

    # Verifica se a API key está configurada
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERRO: OPENROUTER_API_KEY não está configurada")
        print("Configure com: export OPENROUTER_API_KEY='sua-chave'")
        return

    analyzer = UIAnalyzer(
        base_url="http://localhost:80",
        output_dir="/tmp/ui_analysis",
        llm_model="google/gemini-2.5-flash-preview-05-20",
        llm_provider="openrouter"
    )

    # Pode especificar aspectos específicos ou None para todos
    # aspects = ["estado_idle", "execucao_tarefa"]
    aspects = None  # Analisa tudo

    report = await analyzer.analyze(aspects)

    print("\n" + "="*60)
    print("ANÁLISE COMPLETA")
    print("="*60)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
