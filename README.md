# Dumont Shared

Módulos compartilhados entre projetos Dumont.

## Instalação

```bash
pip install -e vendor/dumont-shared
```

## Módulos Disponíveis

### LLM Manager (`dumont.llm`)

Gerenciador unificado de conexões com LLMs:
- OpenRouter (API cloud)
- Ollama local
- Ollama remoto (VPS com GPU via túnel SSH)

```python
from dumont.llm import get_llm_manager

# Criar manager
manager = get_llm_manager()

# Obter LLM (auto-seleciona provider)
llm = await manager.get_llm("auto")

# Usar provider específico
llm = await manager.get_llm("ollama-remote", model="qwen2.5-coder:14b")

# Listar modelos disponíveis
models = await manager.list_models("ollama-remote")
```

## Configuração (`.env`)

```env
# OpenRouter
OPENROUTER_API_KEY=sk-or-...

# Ollama Remoto (VPS com GPU)
OLLAMA_REMOTE_HOST=163.5.212.85
OLLAMA_REMOTE_SSH_PORT=23870
OLLAMA_REMOTE_OLLAMA_PORT=11434
OLLAMA_REMOTE_USER=root
OLLAMA_LOCAL_PORT=11435

# Provider padrão
LLM_PROVIDER=auto
```

## Licença

MIT
