"""Quick-start functions for cloud GPU.

Super simple functions for common operations.

Usage:
    from dumont_core.cloud.quick import start, stop, test, status

    # Start GPU (creates if needed)
    await start()

    # Test the model
    await test("Write hello world in Python")

    # Get status
    status()

    # Stop when done
    await stop()
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx


def status() -> dict:
    """Get current GPU status.

    Returns dict with:
        - status: "running", "paused", or "none"
        - instance_id: int or None
        - gpu_name: str or None
        - cost_per_hour: float
        - model: str
        - ssh_command: str (if running)
        - tunnel_command: str (if running)
    """
    state_file = Path.home() / ".lca" / "gpu_state.json"

    if not state_file.exists():
        return {"status": "none", "instance_id": None}

    try:
        with open(state_file) as f:
            state = json.load(f)
    except json.JSONDecodeError:
        return {"status": "none", "instance_id": None}

    result = {
        "status": state.get("status", "none"),
        "instance_id": state.get("instance_id"),
        "gpu_name": state.get("gpu_name"),
        "cost_per_hour": state.get("cost_per_hour", 0),
        "model": state.get("model_name", "qwen3-coder:30b-a3b-q4_K_M"),
    }

    if state.get("ssh_host") and state.get("ssh_port"):
        result["ssh_command"] = f"ssh -p {state['ssh_port']} root@{state['ssh_host']}"
        result["tunnel_command"] = f"ssh -L 11434:localhost:11434 -p {state['ssh_port']} root@{state['ssh_host']}"

    return result


async def start() -> dict:
    """Start or create GPU instance.

    Creates a new instance if none exists, or resumes a paused one.

    Returns:
        Status dict after starting
    """
    from .gpu_lifecycle import GPULifecycleManager

    manager = GPULifecycleManager()
    await manager.ensure_running()
    return await manager.get_status()


async def stop() -> None:
    """Stop/pause the GPU instance.

    The instance can be resumed later with start().
    """
    from .gpu_lifecycle import GPULifecycleManager

    manager = GPULifecycleManager()
    await manager.pause_instance()


async def destroy() -> None:
    """Destroy the GPU instance completely.

    This cannot be undone. Use stop() to pause instead.
    """
    from .gpu_lifecycle import GPULifecycleManager

    manager = GPULifecycleManager()
    await manager.destroy_instance()


async def test(prompt: str = "Write hello world in Python", model: str | None = None) -> str:
    """Test the model with a prompt.

    Args:
        prompt: The prompt to send
        model: Model name (uses default if not specified)

    Returns:
        The model's response text

    Raises:
        ConnectionError: If cannot connect (tunnel not running)
    """
    if model is None:
        from .config import get_config
        model = get_config().model

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                },
            )

            if resp.status_code != 200:
                raise RuntimeError(f"Model error: {resp.text}")

            return resp.json().get("response", "")

    except httpx.ConnectError:
        raise ConnectionError(
            "Cannot connect to Ollama. "
            "Make sure the SSH tunnel is running:\n"
            "  lca-gpu tunnel"
        )


async def chat(messages: list[dict], model: str | None = None) -> str:
    """Chat with the model.

    Args:
        messages: List of {"role": "user"|"assistant", "content": "..."}
        model: Model name (uses default if not specified)

    Returns:
        The assistant's response

    Example:
        response = await chat([
            {"role": "user", "content": "What is 2+2?"}
        ])
    """
    if model is None:
        from .config import get_config
        model = get_config().model

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                },
            )

            if resp.status_code != 200:
                raise RuntimeError(f"Chat error: {resp.text}")

            return resp.json().get("message", {}).get("content", "")

    except httpx.ConnectError:
        raise ConnectionError(
            "Cannot connect to Ollama. "
            "Make sure the SSH tunnel is running:\n"
            "  lca-gpu tunnel"
        )


async def models() -> list[str]:
    """List installed models on the GPU.

    Returns:
        List of model names
    """
    from .gpu_lifecycle import GPULifecycleManager

    manager = GPULifecycleManager()
    result = await manager.list_models()
    return [m["name"] for m in result]


async def install(model: str) -> None:
    """Install a model on the GPU.

    Args:
        model: Model name (e.g., "qwen3-coder:30b-a3b-q4_K_M")
    """
    import subprocess

    state = status()
    if state["status"] != "running":
        raise RuntimeError("GPU not running. Use start() first.")

    # Parse SSH info from tunnel command
    tunnel_cmd = state.get("tunnel_command", "")
    if not tunnel_cmd:
        raise RuntimeError("No SSH info available")

    # Extract host and port
    parts = tunnel_cmd.split()
    port_idx = parts.index("-p") + 1 if "-p" in parts else None
    port = parts[port_idx] if port_idx else "22"
    host = parts[-1]  # root@host

    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-p", port,
        host,
        f"ollama pull {model}",
    ]

    proc = await asyncio.to_thread(
        subprocess.run, cmd, capture_output=True, text=True
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Install failed: {proc.stderr}")


# Sync wrappers for convenience
def start_sync() -> dict:
    """Synchronous version of start()."""
    return asyncio.run(start())


def stop_sync() -> None:
    """Synchronous version of stop()."""
    asyncio.run(stop())


def test_sync(prompt: str = "Write hello world in Python") -> str:
    """Synchronous version of test()."""
    return asyncio.run(test(prompt))
