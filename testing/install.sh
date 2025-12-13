#!/bin/bash
#
# Dumont Core Testing Module - Instalacao
# ./install.sh         Instala dependencias
# ./install.sh --check Verifica instalacao
#

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

ok() { echo -e "${GREEN}[OK]${NC} $1"; }
err() { echo -e "${RED}[ERRO]${NC} $1"; }

# Diretorio do modulo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

check_install() {
    echo "Verificando instalacao..."
    echo ""

    # Python
    python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null && \
        ok "Python $(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" || \
        { err "Python 3.10+ necessario"; exit 1; }

    # Pacotes
    python3 -c "import browser_use" 2>/dev/null && ok "browser_use" || err "browser_use nao instalado"
    python3 -c "import playwright" 2>/dev/null && ok "playwright" || err "playwright nao instalado"
    python3 -c "import langchain_openai" 2>/dev/null && ok "langchain_openai" || err "langchain_openai nao instalado"

    # Chromium
    python3 -c "from playwright.sync_api import sync_playwright; p = sync_playwright().start(); b = p.chromium.launch(); b.close(); p.stop()" 2>/dev/null && \
        ok "Chromium" || err "Chromium (execute: playwright install chromium)"

    # Modulo
    echo ""
    PYTHONPATH="$MODULE_DIR:$PYTHONPATH" python3 -c "from dumont_core.testing import UnifiedTestRunner" 2>/dev/null && \
        ok "dumont_core.testing" || err "Falha ao importar modulo"

    echo ""
    echo "Variaveis de ambiente:"
    [ -n "$OPENROUTER_API_KEY" ] && ok "OPENROUTER_API_KEY" || echo "   OPENROUTER_API_KEY nao configurada"
    [ -n "$ANTHROPIC_API_KEY" ] && ok "ANTHROPIC_API_KEY" || echo "   ANTHROPIC_API_KEY nao configurada"
}

install() {
    echo "Instalando dependencias..."
    echo ""

    # Verifica Python
    python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null || \
        { err "Python 3.10+ necessario"; exit 1; }

    # Instala pacotes
    pip3 install -q browser-use>=0.10.1 playwright langchain-openai && ok "Pacotes Python instalados"

    # Instala Chromium
    python3 -m playwright install chromium && ok "Chromium instalado"

    echo ""
    ok "Instalacao concluida!"
    echo ""
    echo "Configure uma API key:"
    echo "  export OPENROUTER_API_KEY='sk-or-v1-...'"
}

case "${1:-}" in
    --check|-c) check_install ;;
    --help|-h) echo "Uso: ./install.sh [--check]" ;;
    *) install ;;
esac
