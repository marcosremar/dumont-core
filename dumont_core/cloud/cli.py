#!/usr/bin/env python3
"""Cloud GPU CLI.

Simple command-line interface for managing cloud GPU instances.

Usage:
    lca-gpu status          # Show current instance status
    lca-gpu start           # Start/create GPU instance
    lca-gpu stop            # Stop/pause instance
    lca-gpu destroy         # Destroy instance completely
    lca-gpu ssh             # SSH into instance
    lca-gpu tunnel          # Create SSH tunnel to services
    lca-gpu models          # List installed models
    lca-gpu install MODEL   # Install a model
    lca-gpu config          # Show/edit configuration
    lca-gpu test            # Test model with simple prompt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path


def get_state() -> dict:
    """Load current GPU state."""
    state_file = Path.home() / ".lca" / "gpu_state.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {"status": "none", "instance_id": None}


def get_config() -> dict:
    """Load current config."""
    from .config import get_config as _get_config
    config = _get_config()
    return {
        "model": config.model,
        "max_cost_per_hour": config.max_cost_per_hour,
        "idle_pause_minutes": config.idle_pause_minutes,
        "max_runtime_minutes": config.max_runtime_minutes,
    }


def cmd_status(args):
    """Show instance status."""
    state = get_state()
    config = get_config()

    print("=" * 50)
    print("LCA Cloud GPU Status")
    print("=" * 50)
    print()

    status = state.get("status", "none")
    status_emoji = {
        "running": "\033[92m● Running\033[0m",
        "paused": "\033[93m◐ Paused\033[0m",
        "none": "\033[91m○ Not created\033[0m",
    }.get(status, status)

    print(f"Status:     {status_emoji}")

    if state.get("instance_id"):
        print(f"Instance:   {state['instance_id']}")

    if state.get("gpu_name"):
        print(f"GPU:        {state['gpu_name']}")

    if state.get("cost_per_hour"):
        print(f"Cost:       ${state['cost_per_hour']:.4f}/hr")

    print(f"Model:      {config['model']}")

    if state.get("ssh_host") and state.get("ssh_port"):
        print()
        print("Connection:")
        print(f"  SSH:      ssh -p {state['ssh_port']} root@{state['ssh_host']}")
        print(f"  Tunnel:   ssh -L 11434:localhost:11434 -p {state['ssh_port']} root@{state['ssh_host']}")

    print()


def cmd_start(args):
    """Start or create GPU instance."""
    async def _start():
        from .gpu_lifecycle import GPULifecycleManager
        manager = GPULifecycleManager()
        await manager.ensure_running()
        status = await manager.get_status()
        print(json.dumps(status, indent=2, default=str))

    print("Starting GPU instance...")
    asyncio.run(_start())


def cmd_stop(args):
    """Stop/pause the instance."""
    async def _stop():
        from .gpu_lifecycle import GPULifecycleManager
        manager = GPULifecycleManager()
        await manager.pause_instance()

    print("Stopping GPU instance...")
    asyncio.run(_stop())
    print("Instance stopped.")


def cmd_destroy(args):
    """Destroy the instance completely."""
    state = get_state()
    if state.get("status") == "none":
        print("No instance to destroy.")
        return

    if not args.force:
        response = input(f"Destroy instance {state.get('instance_id')}? [y/N] ")
        if response.lower() != "y":
            print("Cancelled.")
            return

    async def _destroy():
        from .gpu_lifecycle import GPULifecycleManager
        manager = GPULifecycleManager()
        await manager.destroy_instance()

    asyncio.run(_destroy())
    print("Instance destroyed.")


def cmd_ssh(args):
    """SSH into the instance."""
    state = get_state()
    if not state.get("ssh_host") or not state.get("ssh_port"):
        print("No running instance. Use 'lca-gpu start' first.")
        sys.exit(1)

    cmd = ["ssh", "-p", str(state["ssh_port"]), f"root@{state['ssh_host']}"]
    if args.command:
        cmd.append(args.command)

    os.execvp("ssh", cmd)


def cmd_tunnel(args):
    """Create SSH tunnel to services."""
    state = get_state()
    if not state.get("ssh_host") or not state.get("ssh_port"):
        print("No running instance. Use 'lca-gpu start' first.")
        sys.exit(1)

    port = args.port or 11434
    cmd = [
        "ssh",
        "-L", f"{port}:localhost:{port}",
        "-N",
        "-p", str(state["ssh_port"]),
        f"root@{state['ssh_host']}",
    ]

    print(f"Creating tunnel: localhost:{port} -> instance:{port}")
    print("Press Ctrl+C to stop.")
    print()

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nTunnel closed.")


def cmd_models(args):
    """List installed models."""
    state = get_state()
    if state.get("status") != "running":
        print("Instance not running. Use 'lca-gpu start' first.")
        sys.exit(1)

    async def _list_models():
        from .gpu_lifecycle import GPULifecycleManager
        manager = GPULifecycleManager()
        return await manager.list_models()

    models = asyncio.run(_list_models())
    if models:
        print("Installed models:")
        for m in models:
            print(f"  - {m['name']} ({m.get('size', 'unknown')})")
    else:
        print("No models found.")


def cmd_install(args):
    """Install a model."""
    state = get_state()
    if state.get("status") != "running":
        print("Instance not running. Use 'lca-gpu start' first.")
        sys.exit(1)

    model = args.model
    print(f"Installing model: {model}")

    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-p", str(state["ssh_port"]),
        f"root@{state['ssh_host']}",
        f"ollama pull {model}",
    ]

    subprocess.run(cmd)


def cmd_config(args):
    """Show or edit configuration."""
    from .config import CloudConfig, MODELS

    config = CloudConfig.load()

    if args.set_model:
        config.model = args.set_model
        config.save()
        print(f"Model set to: {args.set_model}")
        return

    if args.set_cost:
        config.max_cost_per_hour = float(args.set_cost)
        config.save()
        print(f"Max cost set to: ${args.set_cost}/hr")
        return

    print("Current Configuration:")
    print(f"  Model:           {config.model}")
    print(f"  Max cost:        ${config.max_cost_per_hour}/hr")
    print(f"  Min GPU RAM:     {config.min_gpu_ram_gb} GB")
    print(f"  Idle pause:      {config.idle_pause_minutes} min")
    print(f"  Max runtime:     {config.max_runtime_minutes} min")
    print(f"  Interruptible:   {config.interruptible}")
    print()
    print("Available models:")
    for model_id, info in MODELS.items():
        current = " (current)" if model_id == config.model else ""
        print(f"  - {model_id}{current}")
        print(f"    {info['description']}")


def cmd_test(args):
    """Test the model with a simple prompt."""
    import httpx

    prompt = args.prompt or "Write a hello world function in Python"
    model = args.model

    if not model:
        config = get_config()
        model = config["model"]

    print(f"Testing model: {model}")
    print(f"Prompt: {prompt}")
    print()

    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                },
            )

            if resp.status_code != 200:
                print(f"Error: {resp.text}")
                sys.exit(1)

            data = resp.json()
            print("Response:")
            print(data.get("response", ""))
            print()

            tokens = data.get("eval_count", 0)
            time_ns = data.get("eval_duration", 1)
            time_s = time_ns / 1e9
            tok_per_s = tokens / time_s if time_s > 0 else 0

            print(f"Tokens: {tokens}")
            print(f"Speed:  {tok_per_s:.1f} tok/s")

    except httpx.ConnectError:
        print("Error: Cannot connect to Ollama.")
        print("Make sure the SSH tunnel is running: lca-gpu tunnel")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LCA Cloud GPU Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    lca-gpu status              # Show instance status
    lca-gpu start               # Start/create instance
    lca-gpu tunnel              # Create SSH tunnel
    lca-gpu test "Hello"        # Test model
    lca-gpu config --model X    # Set default model
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status
    subparsers.add_parser("status", help="Show instance status")

    # Start
    subparsers.add_parser("start", help="Start/create GPU instance")

    # Stop
    subparsers.add_parser("stop", help="Stop/pause instance")

    # Destroy
    destroy_p = subparsers.add_parser("destroy", help="Destroy instance")
    destroy_p.add_argument("-f", "--force", action="store_true", help="Skip confirmation")

    # SSH
    ssh_p = subparsers.add_parser("ssh", help="SSH into instance")
    ssh_p.add_argument("command", nargs="?", help="Command to run")

    # Tunnel
    tunnel_p = subparsers.add_parser("tunnel", help="Create SSH tunnel")
    tunnel_p.add_argument("-p", "--port", type=int, help="Local port (default: 11434)")

    # Models
    subparsers.add_parser("models", help="List installed models")

    # Install
    install_p = subparsers.add_parser("install", help="Install a model")
    install_p.add_argument("model", help="Model to install (e.g., qwen3-coder:30b-a3b-q4_K_M)")

    # Config
    config_p = subparsers.add_parser("config", help="Show/edit configuration")
    config_p.add_argument("--model", dest="set_model", help="Set default model")
    config_p.add_argument("--cost", dest="set_cost", help="Set max cost per hour")

    # Test
    test_p = subparsers.add_parser("test", help="Test model")
    test_p.add_argument("prompt", nargs="?", help="Prompt to test")
    test_p.add_argument("-m", "--model", help="Model to use")

    args = parser.parse_args()

    if not args.command:
        # Default to status
        cmd_status(args)
        return

    commands = {
        "status": cmd_status,
        "start": cmd_start,
        "stop": cmd_stop,
        "destroy": cmd_destroy,
        "ssh": cmd_ssh,
        "tunnel": cmd_tunnel,
        "models": cmd_models,
        "install": cmd_install,
        "config": cmd_config,
        "test": cmd_test,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
