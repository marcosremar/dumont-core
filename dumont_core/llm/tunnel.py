#!/usr/bin/env python3
"""
SSH Tunnel - Gerenciamento de túneis SSH para Ollama remoto
============================================================
Permite conexão com servidores Ollama remotos (VPS com GPU) via túnel SSH.
"""

import os
import subprocess
import socket
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import httpx

logger = logging.getLogger("dumont.llm.tunnel")


@dataclass
class RemoteServerConfig:
    """Configuração de servidor remoto para Ollama"""
    host: str
    ssh_port: int = 22
    ollama_port: int = 11434
    local_port: int = 11434  # Porta local para o túnel
    user: str = "root"
    identity_file: Optional[str] = None

    @classmethod
    def from_env(cls) -> Optional["RemoteServerConfig"]:
        """Cria configuração a partir de variáveis de ambiente"""
        host = os.environ.get("OLLAMA_REMOTE_HOST")
        if not host:
            return None

        return cls(
            host=host,
            ssh_port=int(os.environ.get("OLLAMA_REMOTE_SSH_PORT", "22")),
            ollama_port=int(os.environ.get("OLLAMA_REMOTE_OLLAMA_PORT", "11434")),
            local_port=int(os.environ.get("OLLAMA_LOCAL_PORT", "11434")),
            user=os.environ.get("OLLAMA_REMOTE_USER", "root"),
            identity_file=os.environ.get("OLLAMA_REMOTE_KEY"),
        )


@dataclass
class SSHTunnel:
    """Gerencia um túnel SSH para servidor Ollama remoto"""
    config: RemoteServerConfig
    process: Optional[subprocess.Popen] = None

    def is_active(self) -> bool:
        """Verifica se o túnel está ativo"""
        if self.process is None:
            return False
        return self.process.poll() is None

    def start(self) -> bool:
        """Inicia o túnel SSH"""
        if self.is_active():
            logger.info("Túnel SSH já está ativo")
            return True

        # Verificar se a porta local está livre
        if self._is_port_in_use(self.config.local_port):
            logger.warning(f"Porta {self.config.local_port} já está em uso, verificando se é o Ollama...")
            if self._check_ollama_health(f"http://localhost:{self.config.local_port}"):
                logger.info("Ollama já está acessível na porta local")
                return True
            else:
                logger.error(f"Porta {self.config.local_port} em uso por outro serviço")
                return False

        # Construir comando SSH
        ssh_cmd = [
            "ssh",
            "-N",  # Não executar comando remoto
            "-L", f"{self.config.local_port}:localhost:{self.config.ollama_port}",
            "-p", str(self.config.ssh_port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "ExitOnForwardFailure=yes",
        ]

        if self.config.identity_file:
            ssh_cmd.extend(["-i", self.config.identity_file])

        ssh_cmd.append(f"{self.config.user}@{self.config.host}")

        logger.info(f"Iniciando túnel SSH: {' '.join(ssh_cmd)}")

        try:
            self.process = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Esperar um pouco para o túnel estabelecer
            time.sleep(3)

            if not self.is_active():
                stderr = self.process.stderr.read().decode() if self.process.stderr else ""
                logger.error(f"Túnel SSH falhou: {stderr}")
                return False

            # Verificar se o Ollama está acessível
            if self._check_ollama_health(f"http://localhost:{self.config.local_port}"):
                logger.info("Túnel SSH estabelecido com sucesso!")
                return True
            else:
                logger.error("Túnel estabelecido mas Ollama não responde")
                self.stop()
                return False

        except Exception as e:
            logger.error(f"Erro ao criar túnel SSH: {e}")
            return False

    def stop(self):
        """Para o túnel SSH"""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            logger.info("Túnel SSH encerrado")

    def _is_port_in_use(self, port: int) -> bool:
        """Verifica se uma porta está em uso"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def _check_ollama_health(self, base_url: str) -> bool:
        """Verifica se o Ollama está respondendo"""
        try:
            response = httpx.get(f"{base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
