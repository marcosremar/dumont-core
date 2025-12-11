"""
OpenRouter Provider - Conexão com API OpenRouter
"""

import os
from typing import Any, Optional

class OpenRouterProvider:
    """Provider para OpenRouter API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-4o"):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
    
    def is_available(self) -> bool:
        """Verifica se o provider está disponível"""
        return bool(self.api_key)
    
    async def create_llm(self, **kwargs) -> Any:
        """Cria instância do LLM"""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            from browser_use import ChatOpenAI
        
        return ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            **kwargs
        )
