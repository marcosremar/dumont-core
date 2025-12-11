"""GPU Instance Lifecycle Manager.

Manages automatic creation, pause, resume, and destruction of GPU instances.

Features:
- Auto-pause after 15 minutes of inactivity
- Auto-destroy after 2 hours of total runtime
- Auto-create on first use
- Automatic selection of cheapest RTX 3090 interruptible instances
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# State file location
STATE_FILE = Path.home() / ".lca" / "gpu_state.json"


@dataclass
class GPUInstanceState:
    """Persistent state for GPU instance."""

    instance_id: int | None = None
    status: str = "none"  # none, running, paused, destroyed
    gpu_name: str | None = None
    cost_per_hour: float = 0.0

    # Timestamps
    created_at: datetime | None = None
    last_activity: datetime | None = None
    paused_at: datetime | None = None

    # Connection info
    ssh_host: str | None = None
    ssh_port: int | None = None
    public_ip: str | None = None

    # Model info (default: Qwen3-Coder-30B-A3B)
    model_name: str = "qwen3-coder:30b-a3b-q4_K_M"
    model_port: int = 11434

    # Lifecycle settings
    idle_pause_minutes: int = 15
    max_runtime_minutes: int = 180  # 3 hours for larger model

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "instance_id": self.instance_id,
            "status": self.status,
            "gpu_name": self.gpu_name,
            "cost_per_hour": self.cost_per_hour,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "paused_at": self.paused_at.isoformat() if self.paused_at else None,
            "ssh_host": self.ssh_host,
            "ssh_port": self.ssh_port,
            "public_ip": self.public_ip,
            "model_name": self.model_name,
            "model_port": self.model_port,
            "idle_pause_minutes": self.idle_pause_minutes,
            "max_runtime_minutes": self.max_runtime_minutes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GPUInstanceState:
        """Create from dictionary."""
        state = cls()
        state.instance_id = data.get("instance_id")
        state.status = data.get("status", "none")
        state.gpu_name = data.get("gpu_name")
        state.cost_per_hour = data.get("cost_per_hour", 0.0)
        state.ssh_host = data.get("ssh_host")
        state.ssh_port = data.get("ssh_port")
        state.public_ip = data.get("public_ip")
        state.model_name = data.get("model_name", "qwen3-coder:30b-a3b-q4_K_M")
        state.model_port = data.get("model_port", 11434)
        state.idle_pause_minutes = data.get("idle_pause_minutes", 15)
        state.max_runtime_minutes = data.get("max_runtime_minutes", 120)

        if data.get("created_at"):
            state.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("last_activity"):
            state.last_activity = datetime.fromisoformat(data["last_activity"])
        if data.get("paused_at"):
            state.paused_at = datetime.fromisoformat(data["paused_at"])

        return state

    def save(self):
        """Save state to file."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls) -> GPUInstanceState:
        """Load state from file."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    return cls.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError):
                pass
        return cls()

    def should_pause(self) -> bool:
        """Check if instance should be paused due to inactivity."""
        if self.status != "running" or not self.last_activity:
            return False

        idle_time = datetime.now() - self.last_activity
        return idle_time > timedelta(minutes=self.idle_pause_minutes)

    def should_destroy(self) -> bool:
        """Check if instance should be destroyed due to max runtime."""
        if self.status == "none" or not self.created_at:
            return False

        runtime = datetime.now() - self.created_at
        return runtime > timedelta(minutes=self.max_runtime_minutes)

    def get_runtime_minutes(self) -> float:
        """Get total runtime in minutes."""
        if not self.created_at:
            return 0.0
        return (datetime.now() - self.created_at).total_seconds() / 60

    def get_idle_minutes(self) -> float:
        """Get idle time in minutes."""
        if not self.last_activity:
            return 0.0
        return (datetime.now() - self.last_activity).total_seconds() / 60


class GPULifecycleManager:
    """
    Manages GPU instance lifecycle with automatic pause/resume/destroy.

    Usage:
        manager = GPULifecycleManager()

        # Get or create instance (auto-creates if needed)
        instance = await manager.ensure_running()

        # Use the instance...

        # Mark activity to prevent auto-pause
        manager.mark_activity()

        # Check lifecycle
        await manager.check_lifecycle()
    """

    def __init__(self):
        """Initialize the manager."""
        self.state = GPUInstanceState.load()
        self._vastai = None

    @property
    def vastai(self):
        """Lazy load VastAI manager."""
        if self._vastai is None:
            from .vastai import VastAIManager
            self._vastai = VastAIManager()
        return self._vastai

    async def get_status(self) -> dict[str, Any]:
        """Get current instance status."""
        # Refresh state from Vast.ai if we have an instance
        if self.state.instance_id:
            await self._refresh_state()

        result = {
            "status": self.state.status,
            "instance_id": self.state.instance_id,
            "gpu_name": self.state.gpu_name,
            "cost_per_hour": self.state.cost_per_hour,
            "model": self.state.model_name,
            "runtime_minutes": self.state.get_runtime_minutes(),
            "idle_minutes": self.state.get_idle_minutes(),
            "max_runtime_minutes": self.state.max_runtime_minutes,
            "idle_pause_minutes": self.state.idle_pause_minutes,
        }

        if self.state.ssh_host and self.state.ssh_port:
            result["ssh"] = f"ssh -p {self.state.ssh_port} root@{self.state.ssh_host}"
            result["tunnel"] = f"ssh -L {self.state.model_port}:localhost:{self.state.model_port} -p {self.state.ssh_port} root@{self.state.ssh_host}"

        if self.state.status == "running":
            result["time_until_pause"] = max(0, self.state.idle_pause_minutes - self.state.get_idle_minutes())
            result["time_until_destroy"] = max(0, self.state.max_runtime_minutes - self.state.get_runtime_minutes())

        return result

    async def list_models(self) -> list[dict]:
        """List active models on the instance."""
        if self.state.status != "running":
            return []

        # Get models from Ollama
        models = await self._run_ssh_command("ollama list 2>/dev/null || echo ''")

        result = []
        for line in models.strip().split("\n")[1:]:  # Skip header
            if line.strip():
                parts = line.split()
                if parts:
                    result.append({
                        "name": parts[0],
                        "size": parts[2] if len(parts) > 2 else "unknown",
                        "port": self.state.model_port,
                        "endpoint": f"http://localhost:{self.state.model_port}",
                    })

        return result

    async def ensure_running(self) -> GPUInstanceState:
        """Ensure an instance is running, creating one if needed."""
        # Refresh state
        if self.state.instance_id:
            await self._refresh_state()

        if self.state.status == "running":
            self.mark_activity()
            return self.state

        if self.state.status == "paused":
            await self._resume_instance()
            return self.state

        # Need to create new instance
        await self._create_instance()
        return self.state

    async def _create_instance(self):
        """Create a new GPU instance with RTX 3090/4090."""
        from .config import get_config

        config = get_config()
        logger.info(f"Creating new GPU instance for model: {config.model}")

        # Search for GPU instances matching config
        offers = await self.vastai.search_offers(
            min_gpu_ram_gb=config.min_gpu_ram_gb,
            max_cost_per_hour=config.max_cost_per_hour,
            gpu_names=config.preferred_gpus[:2],  # Top 2 preferred
            limit=10,
        )

        if not offers:
            # Fallback to any 24GB+ GPU
            offers = await self.vastai.search_offers(
                min_gpu_ram_gb=config.min_gpu_ram_gb,
                max_cost_per_hour=config.max_cost_per_hour + 0.10,
                gpu_names=config.preferred_gpus,
                limit=10,
            )

        if not offers:
            raise RuntimeError(
                "No suitable GPU instances found. "
                f"Try increasing max_cost_per_hour (current: ${config.max_cost_per_hour}/hr) "
                "or check Vast.ai availability."
            )

        # Pick the cheapest offer
        offer = min(offers, key=lambda o: o.cost_per_hour)

        logger.info(
            f"Selected offer: {offer.gpu_name} @ ${offer.cost_per_hour:.4f}/hr "
            f"({offer.location})"
        )

        # Get model info from config
        model_info = config.get_model_info()

        # Create profile for the configured model
        from .profiles import InstanceProfile, ModelSpec, OLLAMA_QWEN3_CODER_STARTUP, OLLAMA_QWEN_STARTUP

        # Select startup script based on model
        if "qwen3-coder" in config.model:
            startup_script = OLLAMA_QWEN3_CODER_STARTUP
        else:
            startup_script = OLLAMA_QWEN_STARTUP

        profile = InstanceProfile(
            name="lca-gpu",
            description=f"{model_info.get('name', config.model)} via Ollama",
            models=[
                ModelSpec(
                    name=config.model.split(":")[0],
                    model_id=config.model,
                    port=config.model_port,
                    vram_gb=model_info.get("vram_gb", 16),
                    service_type="ollama",
                ),
            ],
            min_gpu_ram_gb=config.min_gpu_ram_gb,
            max_cost_per_hour=config.max_cost_per_hour,
            idle_pause_minutes=config.idle_pause_minutes,
            max_runtime_minutes=config.max_runtime_minutes,
            interruptible=config.interruptible,
            preferred_gpus=config.preferred_gpus,
            startup_script=startup_script,
        )

        # Create instance
        instance = await self.vastai.create_instance(profile, offer)

        # Update state
        self.state.instance_id = instance.id
        self.state.status = "running"
        self.state.gpu_name = instance.gpu_name
        self.state.cost_per_hour = instance.cost_per_hour
        self.state.created_at = datetime.now()
        self.state.last_activity = datetime.now()
        self.state.ssh_host = instance.ssh_host
        self.state.ssh_port = instance.ssh_port
        self.state.public_ip = instance.public_ip
        self.state.save()

        logger.info(f"Instance {instance.id} created and ready!")

    async def _refresh_state(self):
        """Refresh state from Vast.ai."""
        if not self.state.instance_id:
            return

        instance = await self.vastai.get_instance(self.state.instance_id)

        if instance is None:
            # Instance was destroyed externally
            self.state.status = "none"
            self.state.instance_id = None
            self.state.save()
            return

        # Map Vast.ai status to our status
        if instance.status == "running":
            self.state.status = "running"
        elif instance.status in ("stopped", "paused"):
            self.state.status = "paused"
        elif instance.status in ("exited", "destroyed"):
            self.state.status = "none"
            self.state.instance_id = None

        self.state.ssh_host = instance.ssh_host
        self.state.ssh_port = instance.ssh_port
        self.state.public_ip = instance.public_ip
        self.state.save()

    async def _resume_instance(self):
        """Resume a paused instance."""
        if not self.state.instance_id:
            raise RuntimeError("No instance to resume")

        logger.info(f"Resuming instance {self.state.instance_id}...")

        # Start the instance via Vast.ai API
        result = await self.vastai._run_command([
            "start", "instance", str(self.state.instance_id), "--raw"
        ])

        if result.returncode != 0:
            logger.warning(f"Failed to resume, creating new instance: {result.stderr}")
            self.state.status = "none"
            self.state.instance_id = None
            await self._create_instance()
            return

        # Wait for instance to be ready
        for _ in range(60):  # 5 minutes max
            await self._refresh_state()
            if self.state.status == "running":
                break
            await asyncio.sleep(5)

        self.state.last_activity = datetime.now()
        self.state.paused_at = None
        self.state.save()

        logger.info(f"Instance {self.state.instance_id} resumed!")

    async def pause_instance(self):
        """Pause the current instance."""
        if self.state.status != "running" or not self.state.instance_id:
            return

        logger.info(f"Pausing instance {self.state.instance_id}...")

        result = await self.vastai._run_command([
            "stop", "instance", str(self.state.instance_id), "--raw"
        ])

        if result.returncode == 0:
            self.state.status = "paused"
            self.state.paused_at = datetime.now()
            self.state.save()
            logger.info(f"Instance {self.state.instance_id} paused")
        else:
            logger.error(f"Failed to pause instance: {result.stderr}")

    async def destroy_instance(self):
        """Destroy the current instance."""
        if not self.state.instance_id:
            return

        logger.info(f"Destroying instance {self.state.instance_id}...")

        success = await self.vastai.destroy_instance(self.state.instance_id)

        if success:
            self.state.status = "none"
            self.state.instance_id = None
            self.state.created_at = None
            self.state.last_activity = None
            self.state.paused_at = None
            self.state.ssh_host = None
            self.state.ssh_port = None
            self.state.public_ip = None
            self.state.save()
            logger.info("Instance destroyed")
        else:
            logger.error("Failed to destroy instance")

    def mark_activity(self):
        """Mark activity to prevent auto-pause."""
        self.state.last_activity = datetime.now()
        self.state.save()

    async def check_lifecycle(self) -> str | None:
        """
        Check lifecycle and take action if needed.

        Returns:
            Action taken: "paused", "destroyed", or None
        """
        if self.state.status == "none":
            return None

        # Check if should destroy (max runtime exceeded)
        if self.state.should_destroy():
            await self.destroy_instance()
            return "destroyed"

        # Check if should pause (idle timeout)
        if self.state.should_pause():
            await self.pause_instance()
            return "paused"

        return None

    async def _run_ssh_command(self, command: str) -> str:
        """Run a command on the instance via SSH."""
        if not self.state.ssh_host or not self.state.ssh_port:
            return ""

        import subprocess

        ssh_cmd = [
            "ssh",
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-p", str(self.state.ssh_port),
            f"root@{self.state.ssh_host}",
            command,
        ]

        result = await asyncio.to_thread(
            subprocess.run,
            ssh_cmd,
            capture_output=True,
            text=True,
        )

        return result.stdout


# CLI interface
async def cli_main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="GPU Instance Lifecycle Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status command
    subparsers.add_parser("status", help="Show instance status")

    # Models command
    subparsers.add_parser("models", help="List active models")

    # Create command
    subparsers.add_parser("create", help="Create a new instance")

    # Pause command
    subparsers.add_parser("pause", help="Pause the instance")

    # Resume command
    subparsers.add_parser("resume", help="Resume the instance")

    # Destroy command
    subparsers.add_parser("destroy", help="Destroy the instance")

    # Use command (ensure running + mark activity)
    subparsers.add_parser("use", help="Ensure instance is running and mark activity")

    # Check command (check lifecycle)
    subparsers.add_parser("check", help="Check lifecycle and take action if needed")

    args = parser.parse_args()

    manager = GPULifecycleManager()

    if args.command == "status":
        status = await manager.get_status()
        print(json.dumps(status, indent=2, default=str))

    elif args.command == "models":
        models = await manager.list_models()
        if models:
            print(json.dumps(models, indent=2))
        else:
            print("No models running (instance may not be active)")

    elif args.command == "create":
        await manager.ensure_running()
        status = await manager.get_status()
        print(json.dumps(status, indent=2, default=str))

    elif args.command == "pause":
        await manager.pause_instance()
        print("Instance paused")

    elif args.command == "resume":
        await manager.ensure_running()
        print("Instance resumed")

    elif args.command == "destroy":
        await manager.destroy_instance()
        print("Instance destroyed")

    elif args.command == "use":
        await manager.ensure_running()
        manager.mark_activity()
        status = await manager.get_status()
        print(json.dumps(status, indent=2, default=str))

    elif args.command == "check":
        action = await manager.check_lifecycle()
        if action:
            print(f"Action taken: {action}")
        else:
            print("No action needed")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(cli_main())
