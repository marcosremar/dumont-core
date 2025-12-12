"""Cloud GPU configuration.

Centralized configuration for cloud GPU deployments.
Easy to change default model, costs, and behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# =============================================================================
# DEFAULT MODEL CONFIGURATION
# =============================================================================

# The default model to deploy on cloud GPUs
DEFAULT_MODEL = "qwen3-coder:30b-a3b-q4_K_M"
DEFAULT_MODEL_DISPLAY = "Qwen3-Coder-30B-A3B (MoE, Q4_K_M)"

# Model specifications
MODELS = {
    "qwen3-coder:30b-a3b-q4_K_M": {
        "name": "Qwen3-Coder-30B-A3B",
        "description": "MoE model: 30B total params, 3.3B active. Best for agentic coding.",
        "vram_gb": 22,
        "context_length": 32768,
        "quantization": "Q4_K_M",
        "expected_speed": "22-54 tok/s on RTX 3090",
        "service": "ollama",
        "port": 11434,
    },
    "qwen2.5-coder:7b-instruct": {
        "name": "Qwen2.5-Coder-7B",
        "description": "Smaller, faster model for simpler tasks.",
        "vram_gb": 8,
        "context_length": 32768,
        "quantization": None,
        "expected_speed": "~160 tok/s on RTX 4090",
        "service": "ollama",
        "port": 11434,
    },
}

# =============================================================================
# COST & HARDWARE LIMITS
# =============================================================================

# Maximum cost per hour (USD)
MAX_COST_PER_HOUR = 0.40

# Preferred GPUs in order of preference
PREFERRED_GPUS = ["RTX 4090", "RTX 3090", "RTX 3090 Ti", "A6000", "L40"]

# Minimum hardware requirements
MIN_GPU_RAM_GB = 24
MIN_CPU_CORES = 4
MIN_RAM_GB = 32
MIN_DISK_GB = 40

# =============================================================================
# LIFECYCLE SETTINGS
# =============================================================================

# Auto-pause after N minutes of inactivity (0 = disabled)
IDLE_PAUSE_MINUTES = 15

# Auto-destroy after N minutes total runtime (0 = disabled)
MAX_RUNTIME_MINUTES = 180  # 3 hours

# Use interruptible/spot instances (cheaper but can be terminated)
USE_INTERRUPTIBLE = True

# =============================================================================
# PATHS & STATE
# =============================================================================

# State file location
STATE_DIR = Path.home() / ".lca"
STATE_FILE = STATE_DIR / "gpu_state.json"
CONFIG_FILE = STATE_DIR / "gpu_config.json"

# =============================================================================
# DOCKER IMAGE
# =============================================================================

DOCKER_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"


# =============================================================================
# CONFIGURATION CLASS
# =============================================================================

@dataclass
class CloudConfig:
    """Runtime configuration for cloud GPU."""

    # Model
    model: str = DEFAULT_MODEL
    model_port: int = 11434

    # Cost limits
    max_cost_per_hour: float = MAX_COST_PER_HOUR

    # Hardware
    min_gpu_ram_gb: float = MIN_GPU_RAM_GB
    preferred_gpus: list[str] = field(default_factory=lambda: PREFERRED_GPUS.copy())

    # Lifecycle
    idle_pause_minutes: int = IDLE_PAUSE_MINUTES
    max_runtime_minutes: int = MAX_RUNTIME_MINUTES
    interruptible: bool = USE_INTERRUPTIBLE

    @classmethod
    def load(cls) -> "CloudConfig":
        """Load config from file or return defaults."""
        import json
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    data = json.load(f)
                return cls(**data)
            except (json.JSONDecodeError, TypeError):
                pass
        return cls()

    def save(self) -> None:
        """Save config to file."""
        import json
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump({
                "model": self.model,
                "model_port": self.model_port,
                "max_cost_per_hour": self.max_cost_per_hour,
                "min_gpu_ram_gb": self.min_gpu_ram_gb,
                "preferred_gpus": self.preferred_gpus,
                "idle_pause_minutes": self.idle_pause_minutes,
                "max_runtime_minutes": self.max_runtime_minutes,
                "interruptible": self.interruptible,
            }, f, indent=2)

    def get_model_info(self) -> dict:
        """Get info about the configured model."""
        return MODELS.get(self.model, {
            "name": self.model,
            "description": "Custom model",
            "vram_gb": 16,
            "service": "ollama",
            "port": self.model_port,
        })


def get_config() -> CloudConfig:
    """Get the current cloud configuration."""
    return CloudConfig.load()


def set_model(model: str) -> None:
    """Set the default model and save config."""
    config = CloudConfig.load()
    config.model = model
    config.save()


def list_available_models() -> list[dict]:
    """List all available models with their specs."""
    return [
        {"id": model_id, **info}
        for model_id, info in MODELS.items()
    ]
