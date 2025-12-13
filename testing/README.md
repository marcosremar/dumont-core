# Dumont Core - Testing Module

Sistema de testes automatizados com IA para aplicacoes web.
Usa exploracao colaborativa entre Browser-Use e Playwright para descoberta e execucao de testes.

## Instalacao Rapida

```bash
# Dentro do diretorio do modulo
./install.sh

# Ou manualmente
pip install browser-use>=0.10.1 playwright langchain-openai
playwright install chromium
```

## Arquitetura

O sistema opera em duas fases distintas:

```
+===========================================================================+
|                    FASE 1: EXPLORACAO COLABORATIVA                        |
+===========================================================================+
|                                                                           |
|   +--------------------+          +--------------------+                  |
|   |  Browser-Use       |          |  Playwright        |                  |
|   |  Explorer          |          |  Planner Agent     |                  |
|   |  ----------------  |          |  ----------------  |                  |
|   |  - Visao (LLM)     |          |  - DOM analysis    |                  |
|   |  - UX insights     |          |  - Seletores CSS   |                  |
|   |  - Fluxos usuario  |          |  - Accessibility   |                  |
|   |  - Edge cases      |          |  - Network/API     |                  |
|   +---------+----------+          +---------+----------+                  |
|             |                               |                              |
|             +---------------+---------------+                              |
|                             |                                              |
|                             v                                              |
|             +-------------------------------+                              |
|             |      Discovery Merger         |                              |
|             |      -----------------        |                              |
|             |      Consolida:               |                              |
|             |      - Features               |                              |
|             |      - User flows             |                              |
|             |      - Test scenarios         |                              |
|             +---------------+---------------+                              |
|                             |                                              |
|                             v                                              |
|             +-------------------------------+                              |
|             |     Unified Test Plan         |                              |
|             |     (JSON estruturado)        |                              |
|             +-------------------------------+                              |
|                                                                           |
+===========================================================================+
|                    FASE 2: EXECUCAO DE TESTES                             |
+===========================================================================+
|                                                                           |
|             +-------------------------------+                              |
|             |     Unified Test Plan         |                              |
|             +---------------+---------------+                              |
|                             |                                              |
|           +-----------------+-----------------+                            |
|           |                                   |                            |
|           v                                   v                            |
|   +--------------------+          +--------------------+                  |
|   |  Browser-Use       |          |  Playwright        |                  |
|   |  Executor          |          |  Executor          |                  |
|   |  ----------------  |          |  ----------------  |                  |
|   |  - Adaptativo      |          |  - Deterministico  |                  |
|   |  - Julgamento UX   |          |  - CI/CD ready     |                  |
|   |  - Screenshots     |          |  - Metricas        |                  |
|   +---------+----------+          +---------+----------+                  |
|             |                               |                              |
|             +---------------+---------------+                              |
|                             |                                              |
|                             v                                              |
|             +-------------------------------+                              |
|             |    Results Consolidator       |                              |
|             |    ---------------------      |                              |
|             |    - Compara resultados       |                              |
|             |    - Identifica falhas        |                              |
|             |    - Gera relatorio HTML      |                              |
|             +-------------------------------+                              |
|                                                                           |
+===========================================================================+
```

## Estrutura de Arquivos

```
testing/
├── __init__.py                 # Exports publicos (50+ classes)
├── ui_analyzer.py              # UIAnalyzer (classe legada)
├── utils.py                    # Utilitarios (retry, validacao, logging)
├── README.md                   # Esta documentacao
├── install.sh                  # Script de instalacao
│
├── models/                     # Modelos de dados
│   ├── __init__.py
│   ├── discovery.py            # Feature, UserFlow, DiscoveryResult
│   ├── unified_plan.py         # TestStep, TestScenario, UnifiedTestPlan
│   └── test_result.py          # ExecutionResult, ConsolidatedReport
│
├── exploration/                # Fase 1: Exploracao
│   ├── __init__.py
│   ├── base_explorer.py        # BaseExplorer (ABC)
│   ├── browseruse_explorer.py  # Exploracao com Browser-Use + LLM
│   ├── playwright_explorer.py  # Wrapper do Playwright Planner
│   └── discovery_merger.py     # Consolida descobertas
│
├── execution/                  # Fase 2: Execucao
│   ├── __init__.py
│   ├── base_executor.py        # BaseExecutor (ABC)
│   ├── browseruse_executor.py  # Executa com Browser-Use
│   ├── playwright_executor.py  # Executa com Playwright
│   └── results_consolidator.py # Combina resultados
│
└── unified_runner.py           # Orquestrador principal
```

## Uso Rapido

### 1. Execucao Completa (Exploracao + Testes)

```python
from dumont_core.testing import UnifiedTestRunner

runner = UnifiedTestRunner(
    base_url="http://localhost:8080",
    output_dir="/tmp/test_results",
    llm_model="google/gemini-2.5-flash",
    llm_provider="openrouter"
)

# Executa tudo: exploracao + testes
results = await runner.run_full()

print(f"Features descobertas: {len(results.browseruse_result.scenario_results)}")
print(f"Taxa de sucesso: {results.overall_pass_rate}%")
```

### 2. Apenas Exploracao

```python
# Fase 1 - descobrir o que testar
plan = await runner.run_exploration_only(
    explorers=["browseruse", "playwright"],
    exploration_timeout=300
)

# Salva plano para usar depois
plan.save("test_plan.json")

# Exporta como Markdown
plan.export_markdown("test_plan.md")

# Analisa descobertas
for feature in plan.features:
    print(f"- {feature.name}: {feature.description}")
```

### 3. Apenas Execucao (com plano existente)

```python
from dumont_core.testing import UnifiedTestPlan

# Carrega plano existente
plan = UnifiedTestPlan.load("test_plan.json")

# Fase 2 - executa testes
results = await runner.run_execution_only(
    plan=plan,
    executors=["browseruse", "playwright"]
)
```

### 4. API Legada (UIAnalyzer)

```python
from dumont_core.testing import UIAnalyzer

analyzer = UIAnalyzer(
    base_url="http://localhost:8080",
    llm_model="google/gemini-2.5-flash",
    llm_provider="openrouter"
)

# API original ainda funciona
report = await analyzer.analyze(
    aspects=["estado_idle", "fluxo_tarefa"],
    max_steps_per_aspect=10
)
```

## Componentes Principais

### UnifiedTestRunner

Orquestrador principal que coordena exploracao e execucao.

| Metodo | Descricao |
|--------|-----------|
| `run_full()` | Executa exploracao + testes |
| `run_exploration_only()` | Apenas fase de descoberta |
| `run_execution_only()` | Apenas executa testes com plano |
| `explore()` | Alias para run_exploration_only |
| `execute()` | Alias para run_execution_only |

### Modelos de Dados

| Classe | Descricao |
|--------|-----------|
| `Feature` | Uma funcionalidade/elemento descoberto |
| `UserFlow` | Sequencia de acoes do usuario |
| `DiscoveryResult` | Resultado da exploracao |
| `TestStep` | Passo individual de teste |
| `TestScenario` | Cenario completo de teste |
| `UnifiedTestPlan` | Plano de testes consolidado |
| `ExecutionResult` | Resultado de um executor |
| `ConsolidatedReport` | Relatorio final consolidado |

### Enums

| Enum | Valores |
|------|---------|
| `FeaturePriority` | critical, high, medium, low |
| `ElementType` | button, input, link, text, form, modal, menu, etc |
| `ActionType` | navigate, click, type, wait, verify, scroll, hover |
| `StepStatus` | passed, failed, skipped, error |
| `ScenarioStatus` | passed, failed, error, skipped |

### Utilitarios

| Funcao/Classe | Descricao |
|---------------|-----------|
| `get_logger(name)` | Logger configurado |
| `retry_async(max_retries)` | Decorador retry para async |
| `retry_sync(max_retries)` | Decorador retry para sync |
| `validate_url(url)` | Valida e normaliza URL |
| `validate_not_empty(value)` | Valida valor nao vazio |
| `TimingContext` | Context manager para timing |
| `AsyncTimingContext` | Async context manager para timing |
| `sanitize_filename(name)` | Remove caracteres invalidos |

## Configuracao

### Variaveis de Ambiente

```bash
# OpenRouter (recomendado - acesso a varios modelos)
export OPENROUTER_API_KEY="sk-or-v1-..."

# Anthropic (Claude)
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Ollama (local)
export OLLAMA_HOST="http://localhost:11434"
```

### Providers Suportados

| Provider | Modelo Padrao | Custo | Velocidade |
|----------|---------------|-------|------------|
| openrouter | google/gemini-2.5-flash | $ | Rapido |
| anthropic | claude-3-5-sonnet-20241022 | $$$ | Medio |
| openai | gpt-4o | $$ | Rapido |
| ollama | llama3.2-vision | Gratis | Lento |

### Opcoes do Runner

```python
runner = UnifiedTestRunner(
    base_url="http://localhost:8080",  # URL da aplicacao
    output_dir="/tmp/results",         # Diretorio para outputs
    headless=True,                     # Browser sem interface
    timeout=60,                        # Timeout por cenario (s)
    llm_model="google/gemini-2.5-flash",
    llm_provider="openrouter",
    max_steps_per_scenario=20,         # Limite de passos
)
```

## Exemplos Avancados

### Teste de Regressao Visual

```python
runner = UnifiedTestRunner(base_url="http://localhost:8080")
results = await runner.run_full()

if results.overall_pass_rate < 100:
    print("REGRESSAO DETECTADA!")
    for scenario in results.browseruse_result.scenario_results:
        if scenario.status != ScenarioStatus.PASSED:
            print(f"  FALHA: {scenario.scenario_name}")
            print(f"         {scenario.ux_summary}")
```

### Geracao de Documentacao de Testes

```python
runner = UnifiedTestRunner(base_url="https://meuapp.com")
plan = await runner.run_exploration_only()

# Exporta plano legivel
plan.export_markdown("docs/test_plan.md")

# Lista todas as features
for f in plan.features:
    print(f"## {f.name}")
    print(f"- Tipo: {f.element_type.value}")
    print(f"- Prioridade: {f.priority.value}")
    if f.selector:
        print(f"- Seletor: `{f.selector}`")
    if f.description:
        print(f"- Descricao: {f.description}")
```

### Integracao com CI/CD (GitHub Actions)

```yaml
# .github/workflows/ui-tests.yml
name: UI Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install browser-use playwright langchain-openai
          playwright install chromium

      - name: Start application
        run: |
          docker-compose up -d
          sleep 10

      - name: Run UI Tests
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: |
          python -c "
          import asyncio
          from dumont_core.testing import UnifiedTestRunner

          async def main():
              runner = UnifiedTestRunner(
                  base_url='http://localhost:8080',
                  output_dir='./test-results'
              )
              results = await runner.run_full()

              # Exporta relatorio
              results.export_html('test-results/report.html')

              # Falha se taxa < 100%
              if results.overall_pass_rate < 100:
                  print(f'Taxa de sucesso: {results.overall_pass_rate}%')
                  exit(1)

          asyncio.run(main())
          "

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: test-results/
```

### Teste Customizado com Steps Manuais

```python
from dumont_core.testing import (
    UnifiedTestPlan, TestScenario, TestStep,
    ActionType, FeaturePriority
)
from datetime import datetime

# Cria plano manualmente
plan = UnifiedTestPlan(
    app_name="Minha App",
    base_url="http://localhost:8080",
    exploration_timestamp=datetime.now(),
    test_scenarios=[
        TestScenario(
            id="login-flow",
            name="Fluxo de Login",
            priority=FeaturePriority.CRITICAL,
            expected_result="Usuario logado com sucesso",
            steps=[
                TestStep(
                    action=ActionType.NAVIGATE,
                    target="/login",
                    description="Abre pagina de login"
                ),
                TestStep(
                    action=ActionType.TYPE,
                    target="input[name='email']",
                    value="user@test.com",
                    description="Digita email"
                ),
                TestStep(
                    action=ActionType.TYPE,
                    target="input[name='password']",
                    value="senha123",
                    description="Digita senha"
                ),
                TestStep(
                    action=ActionType.CLICK,
                    target="button[type='submit']",
                    description="Clica em Entrar"
                ),
                TestStep(
                    action=ActionType.VERIFY,
                    target="Dashboard",
                    condition="text visible",
                    description="Verifica dashboard"
                ),
            ]
        )
    ]
)

# Executa plano customizado
runner = UnifiedTestRunner(base_url="http://localhost:8080")
results = await runner.run_execution_only(plan)
```

## Dependencias

### Obrigatorias

```
browser-use>=0.10.1
playwright>=1.40.0
langchain-openai>=0.1.0
```

### Opcionais

```
langchain-anthropic>=0.1.0    # Para usar Claude
langchain-google-genai>=1.0.0 # Para Gemini direto
```

### Instalacao Completa

```bash
# Dependencias Python
pip install browser-use>=0.10.1 playwright langchain-openai

# Browser do Playwright
playwright install chromium

# Opcional: para testes visuais
pip install pillow opencv-python
```

## Troubleshooting

### Erro: "Browser not found"
```bash
playwright install chromium
```

### Erro: "API key not set"
```bash
export OPENROUTER_API_KEY="sua-chave"
```

### Erro: "Timeout waiting for element"
Aumente o timeout:
```python
runner = UnifiedTestRunner(timeout=120)
```

### Logs detalhados
```python
import logging
logging.getLogger("dumont_testing").setLevel(logging.DEBUG)
```

## Contribuindo

1. Fork o repositorio
2. Crie branch: `git checkout -b feature/nova-funcionalidade`
3. Commit: `git commit -m 'Add nova funcionalidade'`
4. Push: `git push origin feature/nova-funcionalidade`
5. Abra Pull Request

## Licenca

MIT License - veja LICENSE para detalhes.

## Changelog

### v1.0.0
- Arquitetura de duas fases (exploracao + execucao)
- Browser-Use e Playwright explorers
- Discovery Merger para consolidar descobertas
- Executores paralelos
- Results Consolidator com relatorio HTML
- Utilitarios: retry, validacao, timing
- Compatibilidade com UIAnalyzer legado
