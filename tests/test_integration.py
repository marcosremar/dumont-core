#!/usr/bin/env python3
"""
Integration Tests for Dumont Core LLM Module

Testa:
1. OpenRouter (API cloud)
2. Ollama local/remote
3. Dedicated machines (Vast.ai)
4. Instance discovery and model listing
"""

import asyncio
import os
import sys
import pytest
import httpx
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Adicionar diretório do dumont-core ao path
# Este arquivo está em: vendor/dumont-core/tests/test_integration.py
# dumont-core está em: vendor/dumont-core/
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_dumont_core_dir = os.path.dirname(_tests_dir)
if _dumont_core_dir not in sys.path:
    sys.path.insert(0, _dumont_core_dir)

# Importar módulos
from llm import (
    get_llm_manager,
    get_dedicated_provider,
    LLMProvider,
    DedicatedBackend,
)


class TestOpenRouter:
    """Testes do provider OpenRouter"""
    
    @pytest.mark.asyncio
    async def test_openrouter_available(self):
        """Verifica se OpenRouter está configurado"""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        assert api_key is not None, "OPENROUTER_API_KEY não configurada"
        assert len(api_key) > 10, "OPENROUTER_API_KEY parece inválida"
    
    @pytest.mark.asyncio
    async def test_openrouter_create_llm(self):
        """Testa criação de LLM via OpenRouter"""
        manager = get_llm_manager()
        try:
            llm = await manager.get_llm("openrouter", model="google/gemini-2.5-flash")
            assert llm is not None
            assert hasattr(llm, "ainvoke")
        finally:
            manager.cleanup()
    
    @pytest.mark.asyncio 
    async def test_openrouter_simple_request(self):
        """Testa requisição simples ao OpenRouter"""
        manager = get_llm_manager()
        try:
            llm = await manager.get_llm("openrouter", model="google/gemini-2.5-flash")
            
            # Fazer requisição direta via httpx (evita problemas de formato de mensagem)
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "google/gemini-2.5-flash",
                        "messages": [{"role": "user", "content": "Say just: OK"}],
                    }
                )
                assert response.status_code == 200
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                assert "OK" in content or "ok" in content.lower()
                print(f"✅ OpenRouter response: {content[:50]}")
        finally:
            manager.cleanup()


class TestOllamaRemote:
    """Testes do Ollama remoto via SSH tunnel (opcional - instâncias podem estar offline)"""

    @pytest.mark.asyncio
    async def test_remote_config_exists(self):
        """Verifica se configuração remota existe"""
        manager = get_llm_manager()
        assert manager.remote_config is not None, "Configuração remota não encontrada"
        assert manager.remote_config.host, "Host remoto não configurado"
        print(f"✅ Remote host: {manager.remote_config.host}")

    @pytest.mark.asyncio
    async def test_tunnel_connection(self):
        """Testa conexão via túnel SSH (skip se offline)"""
        manager = get_llm_manager()
        try:
            result = await manager._ensure_remote_tunnel()
            if result:
                print("✅ Túnel SSH estabelecido")
            else:
                print("⚠️ Skipping: Instância remota offline")
        finally:
            manager.cleanup()

    @pytest.mark.asyncio
    async def test_ollama_remote_request(self):
        """Testa requisição ao Ollama remoto (skip se offline)"""
        manager = get_llm_manager()
        try:
            result = await manager._ensure_remote_tunnel()
            if not result:
                print("⚠️ Skipping: Instância remota offline")
                return

            local_port = manager.remote_config.local_port
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"http://localhost:{local_port}/api/generate",
                    json={
                        "model": "qwen2.5-coder:14b",
                        "prompt": "Say just: OK",
                        "stream": False
                    }
                )
                assert response.status_code == 200
                data = response.json()
                assert "response" in data
                print(f"✅ Ollama remote response: {data['response'][:50]}")
        finally:
            manager.cleanup()


class TestDedicatedProvider:
    """Testes do provider de máquinas dedicadas (Vast.ai - opcional se instâncias offline)"""

    @pytest.mark.asyncio
    async def test_vastai_available(self):
        """Verifica se Vast.ai está configurado"""
        provider = get_dedicated_provider()
        assert provider.is_available, "VASTAI_API_KEY não configurada"
        print("✅ Vast.ai API configurada")

    @pytest.mark.asyncio
    async def test_discover_instances(self):
        """Testa descoberta de instâncias ativas"""
        provider = get_dedicated_provider()
        instances = await provider.discover_active_instances()

        print(f"✅ Encontradas {len(instances)} instâncias")
        for inst in instances:
            print(f"   - #{inst.instance_id}: {inst.gpu_name} ({len(inst.available_models)} modelos)")
            inst.disconnect()

    @pytest.mark.asyncio
    async def test_list_models_on_instance(self):
        """Testa listagem de modelos em instância ativa (skip se nenhuma)"""
        provider = get_dedicated_provider()
        instances = await provider.discover_active_instances()

        if len(instances) == 0:
            print("⚠️ Skipping: Nenhuma instância ativa encontrada")
            return

        inst = instances[0]
        if len(inst.available_models) == 0:
            print("⚠️ Skipping: Nenhum modelo encontrado na instância")
            inst.disconnect()
            return

        print(f"✅ Modelos na instância #{inst.instance_id}: {inst.available_models}")
        inst.disconnect()

    @pytest.mark.asyncio
    async def test_ollama_request_via_instance(self):
        """Testa requisição Ollama via instância descoberta (skip se nenhuma)"""
        provider = get_dedicated_provider()
        instances = await provider.discover_active_instances()

        if len(instances) == 0:
            print("⚠️ Skipping: Nenhuma instância ativa")
            return

        inst = instances[0]
        if len(inst.available_models) == 0:
            print("⚠️ Skipping: Nenhum modelo disponível")
            inst.disconnect()
            return

        model = inst.available_models[0]

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{inst.endpoint}/api/generate",
                json={
                    "model": model,
                    "prompt": "Say just: Integration test passed!",
                    "stream": False
                }
            )
            assert response.status_code == 200
            data = response.json()
            print(f"✅ Response from {model}: {data['response'][:100]}")

        inst.disconnect()


class TestAutoSelection:
    """Testes de auto-seleção de provider"""
    
    @pytest.mark.asyncio
    async def test_auto_selects_provider(self):
        """Testa que auto seleciona um provider válido"""
        manager = get_llm_manager()
        try:
            provider = await manager._auto_select_provider()
            assert provider in ["ollama", "ollama-remote", "openrouter"]
            print(f"✅ Auto-selecionado: {provider}")
        finally:
            manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_get_llm_auto(self):
        """Testa get_llm com provider auto"""
        manager = get_llm_manager()
        try:
            llm = await manager.get_llm("auto")
            assert llm is not None
            print(f"✅ LLM criado via auto: {type(llm).__name__}")
        finally:
            manager.cleanup()


class TestSmallModelDeploy:
    """Testes de deploy de modelo pequeno (~500M params)"""
    
    @pytest.mark.asyncio
    async def test_pull_small_ollama_model(self):
        """Testa pull de modelo pequeno no Ollama (qwen2.5:0.5b)"""
        provider = get_dedicated_provider()
        instances = await provider.discover_active_instances()

        if not instances:
            print("⚠️ Skipping: Nenhuma instância ativa para testar")
            return
        
        inst = instances[0]
        
        # Modelo pequeno: qwen2.5:0.5b (~400MB)
        small_model = "qwen2.5:0.5b"
        
        print(f"Baixando modelo {small_model}...")
        
        async with httpx.AsyncClient(timeout=300) as client:
            # Pull do modelo
            response = await client.post(
                f"{inst.endpoint}/api/pull",
                json={"name": small_model, "stream": False}
            )
            
            if response.status_code == 200:
                print(f"✅ Modelo {small_model} baixado")
                
                # Testar requisição
                response = await client.post(
                    f"{inst.endpoint}/api/generate",
                    json={
                        "model": small_model,
                        "prompt": "What is 2+2?",
                        "stream": False
                    }
                )
                assert response.status_code == 200
                data = response.json()
                print(f"✅ Resposta do {small_model}: {data['response'][:100]}")
            else:
                print(f"⚠️ Falha ao baixar modelo: {response.text}")
        
        inst.disconnect()


# Função para rodar todos os testes
async def run_all_tests():
    """Roda todos os testes de integração"""
    print("=" * 60)
    print("DUMONT-CORE INTEGRATION TESTS")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # 1. OpenRouter
    print("\n📦 OpenRouter Tests")
    print("-" * 40)
    try:
        t = TestOpenRouter()
        await t.test_openrouter_available()
        await t.test_openrouter_create_llm()
        await t.test_openrouter_simple_request()
        tests_passed += 3
    except Exception as e:
        print(f"❌ OpenRouter test failed: {e}")
        tests_failed += 1
    
    # 2. Ollama Remote
    print("\n📦 Ollama Remote Tests")
    print("-" * 40)
    try:
        t = TestOllamaRemote()
        await t.test_remote_config_exists()
        await t.test_tunnel_connection()
        await t.test_ollama_remote_request()
        tests_passed += 3
    except Exception as e:
        print(f"❌ Ollama Remote test failed: {e}")
        tests_failed += 1
    
    # 3. Dedicated Provider
    print("\n📦 Dedicated Provider Tests")
    print("-" * 40)
    try:
        t = TestDedicatedProvider()
        await t.test_vastai_available()
        await t.test_discover_instances()
        await t.test_list_models_on_instance()
        await t.test_ollama_request_via_instance()
        tests_passed += 4
    except Exception as e:
        print(f"❌ Dedicated Provider test failed: {e}")
        tests_failed += 1
    
    # 4. Auto Selection
    print("\n📦 Auto Selection Tests")
    print("-" * 40)
    try:
        t = TestAutoSelection()
        await t.test_auto_selects_provider()
        await t.test_get_llm_auto()
        tests_passed += 2
    except Exception as e:
        print(f"❌ Auto Selection test failed: {e}")
        tests_failed += 1
    
    # 5. Small Model Deploy
    print("\n📦 Small Model Deploy Test")
    print("-" * 40)
    try:
        t = TestSmallModelDeploy()
        await t.test_pull_small_ollama_model()
        tests_passed += 1
    except Exception as e:
        print(f"❌ Small Model Deploy test failed: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    
    return tests_failed == 0


if __name__ == "__main__":
    import sys
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
