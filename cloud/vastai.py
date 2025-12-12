"""Vast.ai provider for remote LLMLingua-2 compression.

Vast.ai offers very cheap GPU instances (as low as $0.03/hour).
This module handles VM provisioning and management.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VastInstance:
    """Vast.ai instance information."""

    id: int
    status: str  # "running", "loading", "stopped", "exited"
    ip: str | None
    port: int | None
    gpu_name: str
    cpu_cores: float
    ram_gb: float
    disk_gb: float
    cost_per_hour: float
    created_at: datetime | None


@dataclass
class VastConfig:
    """Configuration for Vast.ai provider."""

    api_key: str | None = None

    # Instance requirements
    min_cpu_cores: int = 2
    min_ram_gb: int = 4
    min_disk_gb: int = 10
    min_inet_up: int = 100  # Mbps
    min_reliability: float = 0.9

    # Cost limits
    max_cost_per_hour: float = 0.10  # $0.10/hr max

    # Docker image for LLMLingua service (PyTorch 2.5 for transformers compatibility)
    docker_image: str = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"

    # Service port
    service_port: int = 8080


class VastAIProvider:
    """
    Vast.ai provider for remote compression service.

    Manages lifecycle of cheap GPU instances on Vast.ai.
    """

    def __init__(self, config: VastConfig | None = None):
        """Initialize provider."""
        self.config = config or VastConfig()
        self._api_key = self.config.api_key or os.environ.get("VASTAI_API_KEY")
        self._instance: VastInstance | None = None

        if not self._api_key:
            raise ValueError("VASTAI_API_KEY not set")

    async def find_cheapest_offer(self) -> dict[str, Any] | None:
        """Find the cheapest offer matching requirements."""
        query = (
            f"cpu_cores>={self.config.min_cpu_cores} "
            f"cpu_ram>={self.config.min_ram_gb} "
            f"disk_space>={self.config.min_disk_gb} "
            f"inet_up>={self.config.min_inet_up} "
            f"reliability>{self.config.min_reliability} "
            f"dph_total<={self.config.max_cost_per_hour}"
        )

        result = await self._run_vastai_command([
            "search", "offers", query,
            "--order", "dph_total",
            "--limit", "1",
            "--raw"
        ])

        if result.returncode != 0:
            logger.error(f"Failed to search offers: {result.stderr}")
            return None

        try:
            offers = json.loads(result.stdout)
            if offers and isinstance(offers, list) and len(offers) > 0:
                return offers[0]
        except (json.JSONDecodeError, TypeError, IndexError):
            logger.warning("Failed to parse offers from Vast.ai API response")
            pass

        return None

    async def create_instance(self, offer_id: int | None = None) -> VastInstance:
        """
        Create a new Vast.ai instance.

        Args:
            offer_id: Specific offer ID, or None to find cheapest

        Returns:
            VastInstance with connection details
        """
        if offer_id is None:
            offer = await self.find_cheapest_offer()
            if not offer:
                raise RuntimeError("No suitable offers found")
            offer_id = offer["id"]
            logger.info(f"Selected offer {offer_id}: ${offer['dph_total']:.4f}/hr, {offer['gpu_name']}")

        # Create instance with LLMLingua setup script
        onstart_script = self._get_onstart_script()

        result = await self._run_vastai_command([
            "create", "instance", str(offer_id),
            "--image", self.config.docker_image,
            "--disk", str(self.config.min_disk_gb),
            "--onstart-cmd", onstart_script,
            "--raw"
        ])

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create instance: {result.stderr}")

        # Parse instance ID from response
        # vastai returns text like "Started. {'success': True, 'new_contract': 12345}"
        # or JSON like {"success": true, "new_contract": 12345}
        instance_id = None
        output = result.stdout.strip()

        # Try JSON first
        try:
            data = json.loads(output)
            instance_id = data.get("new_contract")
        except json.JSONDecodeError:
            # Try to extract from text output
            import re
            match = re.search(r"new_contract['\"]?\s*:\s*(\d+)", output)
            if match:
                instance_id = int(match.group(1))

        if not instance_id:
            # Check if an instance was actually created by looking at stderr
            stderr = result.stderr.strip() if result.stderr else ""
            logger.warning(f"Could not parse instance ID. stdout={output}, stderr={stderr}")
            raise RuntimeError(f"No instance ID in response: {output}")

        logger.info(f"Created instance {instance_id}")

        # Wait for instance to be ready
        return await self._wait_for_instance(instance_id)

    async def get_instance(self, instance_id: int) -> VastInstance | None:
        """Get instance details."""
        result = await self._run_vastai_command([
            "show", "instance", str(instance_id), "--raw"
        ])

        if result.returncode != 0:
            return None

        try:
            data = json.loads(result.stdout)
            if not data:
                return None

            instance_data = data[0] if isinstance(data, list) else data

            # Extract SSH/port info
            ip = None
            port = None
            if instance_data.get("public_ipaddr"):
                ip = instance_data["public_ipaddr"]
                # Find the mapped port for our service
                ports = instance_data.get("ports", {})
                port_key = f"{self.config.service_port}/tcp"
                if port_key in ports:
                    port_info = ports[port_key]
                    if isinstance(port_info, list) and port_info:
                        port = int(port_info[0].get("HostPort", self.config.service_port))

            return VastInstance(
                id=instance_data["id"],
                status=instance_data.get("actual_status", "unknown"),
                ip=ip,
                port=port,
                gpu_name=instance_data.get("gpu_name", "unknown"),
                cpu_cores=instance_data.get("cpu_cores_effective", 0),
                ram_gb=instance_data.get("cpu_ram", 0),
                disk_gb=instance_data.get("disk_space", 0),
                cost_per_hour=instance_data.get("dph_total", 0),
                created_at=None,
            )

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse instance data: {e}")
            return None

    async def destroy_instance(self, instance_id: int) -> bool:
        """Destroy an instance."""
        result = await self._run_vastai_command([
            "destroy", "instance", str(instance_id), "--raw"
        ])

        if result.returncode == 0:
            logger.info(f"Destroyed instance {instance_id}")
            self._instance = None
            return True

        logger.error(f"Failed to destroy instance: {result.stderr}")
        return False

    async def list_instances(self) -> list[VastInstance]:
        """List all running instances."""
        result = await self._run_vastai_command([
            "show", "instances", "--raw"
        ])

        if result.returncode != 0:
            return []

        try:
            data = json.loads(result.stdout)
            instances = []

            for item in data:
                instances.append(VastInstance(
                    id=item["id"],
                    status=item.get("actual_status", "unknown"),
                    ip=item.get("public_ipaddr"),
                    port=None,
                    gpu_name=item.get("gpu_name", "unknown"),
                    cpu_cores=item.get("cpu_cores_effective", 0),
                    ram_gb=item.get("cpu_ram", 0),
                    disk_gb=item.get("disk_space", 0),
                    cost_per_hour=item.get("dph_total", 0),
                    created_at=None,
                ))

            return instances

        except json.JSONDecodeError:
            return []

    async def _wait_for_instance(
        self,
        instance_id: int,
        timeout: int = 300
    ) -> VastInstance:
        """Wait for instance to be ready."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            instance = await self.get_instance(instance_id)

            if instance and instance.status == "running":
                if instance.ip and instance.port:
                    logger.info(f"Instance ready at {instance.ip}:{instance.port}")
                    self._instance = instance
                    return instance
                elif instance.ip:
                    # Port mapping might take a moment
                    logger.info(f"Instance running, waiting for port mapping...")

            await asyncio.sleep(5)

        raise RuntimeError(f"Timeout waiting for instance {instance_id}")

    def _get_onstart_script(self) -> str:
        """Get the startup script for the instance."""
        return '''
#!/bin/bash
set -e

# Install dependencies
pip install llmlingua fastapi uvicorn[standard] httpx

# Pre-download model
python -c "
from llmlingua import PromptCompressor
print('Downloading LLMLingua-2 model...')
compressor = PromptCompressor(
    model_name='microsoft/llmlingua-2-xlm-roberta-large-meetingbank',
    use_llmlingua2=True,
    device_map='auto'
)
print('Model ready!')
"

# Create service
cat > /root/llmlingua_service.py << 'PYEOF'
import os
import time
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

_compressor = None

def get_compressor():
    global _compressor
    if _compressor is None:
        from llmlingua import PromptCompressor
        _compressor = PromptCompressor(
            model_name='microsoft/llmlingua-2-xlm-roberta-large-meetingbank',
            use_llmlingua2=True,
            device_map='auto'
        )
    return _compressor

app = FastAPI(title="LLMLingua-2 Service")

class CompressRequest(BaseModel):
    text: str
    target_ratio: float = 0.5
    force_tokens: list[str] = []

class CompressResponse(BaseModel):
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    time_seconds: float

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/compress", response_model=CompressResponse)
def compress(request: CompressRequest):
    start_time = time.time()
    compressor = get_compressor()

    force_tokens = request.force_tokens or [
        '\\n', '```', 'def ', 'class ', 'import ', 'from ',
        'return ', 'if ', 'else:', 'elif ', 'for ', 'while ',
    ]

    result = compressor.compress_prompt(
        request.text,
        rate=request.target_ratio,
        force_tokens=force_tokens,
        drop_consecutive=True,
    )

    compressed_text = result['compressed_prompt']
    original_tokens = len(request.text.split())
    compressed_tokens = len(compressed_text.split())

    return CompressResponse(
        compressed_text=compressed_text,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
        time_seconds=time.time() - start_time
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
PYEOF

# Start service
python /root/llmlingua_service.py
'''

    async def _run_vastai_command(
        self,
        args: list[str]
    ) -> subprocess.CompletedProcess:
        """Run a vastai CLI command."""
        # Try to find vastai in common locations
        vastai_paths = [
            "vastai",  # System PATH
            os.path.expanduser("~/.local/bin/vastai"),  # User local
            "/home/ubuntu/code-agent-2/venv/bin/vastai",  # Project venv
        ]

        vastai_cmd = "vastai"
        for path in vastai_paths:
            if os.path.exists(path):
                vastai_cmd = path
                break

        cmd = [vastai_cmd, "--api-key", self._api_key] + args

        logger.debug(f"Running: vastai {' '.join(args)}")

        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True
        )

        return result


# Singleton (thread-safe)
import threading

_provider: VastAIProvider | None = None
_provider_lock: threading.Lock = threading.Lock()


def get_vastai_provider(config: VastConfig | None = None) -> VastAIProvider:
    """Get or create the Vast.ai provider instance (thread-safe)."""
    global _provider

    if _provider is None:
        with _provider_lock:
            # Double-check locking pattern
            if _provider is None:
                _provider = VastAIProvider(config)

    return _provider
