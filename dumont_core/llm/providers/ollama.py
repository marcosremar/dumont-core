"""
Ollama Provider - Conexão com Ollama local ou remoto
"""

from typing import Any, Optional
import httpx

class OllamaProvider:
    """Provider para Ollama (local ou remoto)"""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen2.5-coder:14b"):
        self.host = host
        self.model = model
    
    def is_available(self) -> bool:
        """Verifica se o Ollama está disponível"""
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> list:
        """Lista modelos disponíveis"""
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=10)
            data = response.json()
            return data.get("models", [])
        except Exception:
            return []
    
    async def create_llm(self, **kwargs) -> Any:
        """Cria instância do LLM"""
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from browser_use import ChatOllama
        
        return ChatOllama(
            model=self.model,
            base_url=self.host,
            timeout=kwargs.pop("timeout", 120),
            **kwargs
        )
