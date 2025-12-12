"""Instance profiles for deploying models on GPU instances.

Profiles define:
- Which models to run
- Hardware requirements
- Startup configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelSpec:
    """Specification for a model to deploy."""

    name: str  # Unique identifier for the model
    model_id: str  # HuggingFace/model path
    port: int  # Port to serve on
    vram_gb: float  # Estimated VRAM usage
    service_type: str = "vllm"  # "vllm", "llmlingua", "ollama", "custom"
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class InstanceProfile:
    """Profile defining an instance configuration."""

    name: str
    description: str
    models: list[ModelSpec]

    # Hardware requirements
    min_gpu_ram_gb: float = 0  # Auto-calculated if 0
    min_cpu_cores: int = 4
    min_ram_gb: int = 16
    min_disk_gb: int = 30
    max_cost_per_hour: float = 0.50

    # Lifecycle settings
    idle_pause_minutes: int = 0  # Auto-pause after N minutes idle (0 = disabled)
    max_runtime_minutes: int = 0  # Auto-destroy after N minutes (0 = disabled)
    interruptible: bool = True  # Use spot/interruptible instances

    # Preferences
    preferred_gpus: list[str] = field(default_factory=lambda: [
        "RTX 4090", "A100", "A6000", "RTX 3090", "L40", "A5000"
    ])

    # Docker
    docker_image: str = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"

    # Startup script (optional, for custom setup)
    startup_script: str | None = None

    def __post_init__(self):
        """Auto-calculate GPU RAM requirement."""
        if self.min_gpu_ram_gb == 0 and self.models:
            # Sum of all model VRAM + 2GB overhead
            self.min_gpu_ram_gb = sum(m.vram_gb for m in self.models) + 2

    def get_ports(self) -> list[int]:
        """Get all ports used by models."""
        return [m.port for m in self.models]

    def get_model(self, name: str) -> ModelSpec | None:
        """Get model spec by name."""
        for model in self.models:
            if model.name == name:
                return model
        return None


# Startup script for Ollama with Qwen2.5-Coder-7B (legacy)
OLLAMA_QWEN_STARTUP = '''#!/bin/bash
set -e

echo "=== Installing Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh

echo "=== Starting Ollama ==="
ollama serve &> /var/log/ollama.log &
sleep 5

echo "=== Pulling Qwen2.5-Coder:7b-instruct ==="
ollama pull qwen2.5-coder:7b-instruct

echo "=== Ready! ==="
echo "Ollama running on port 11434"
echo "Model: qwen2.5-coder:7b-instruct"

# Keep alive
tail -f /var/log/ollama.log
'''

# Startup script for Ollama with Qwen3-Coder-30B-A3B (MoE, RECOMMENDED)
OLLAMA_QWEN3_CODER_STARTUP = '''#!/bin/bash
set -e

echo "=== Installing Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh

echo "=== Starting Ollama ==="
ollama serve &> /var/log/ollama.log &
sleep 5

echo "=== Pulling Qwen3-Coder-30B-A3B (Q4_K_M quantization) ==="
echo "This is a MoE model: 30B total params, 3.3B active"
echo "Expected VRAM: ~20-24GB with Q4_K_M"
ollama pull qwen3-coder:30b-a3b-q4_K_M

echo "=== Ready! ==="
echo "Ollama running on port 11434"
echo "Model: qwen3-coder:30b-a3b-q4_K_M"
echo "Context: up to 32K tokens"
echo "Expected speed: ~22-54 tok/s on RTX 3090"

# Keep alive
tail -f /var/log/ollama.log
'''

# Pre-defined profiles
PROFILES: dict[str, InstanceProfile] = {
    # Qwen3-Coder-30B-A3B via Ollama (RECOMMENDED - Best for agentic coding)
    "qwen3-coder": InstanceProfile(
        name="qwen3-coder",
        description="Qwen3-Coder-30B-A3B MoE via Ollama (30B total, 3.3B active, ~22-54 tok/s)",
        models=[
            ModelSpec(
                name="qwen3-coder-30b",
                model_id="qwen3-coder:30b-a3b-q4_K_M",
                port=11434,
                vram_gb=22,  # Q4_K_M uses ~20-24GB VRAM
                service_type="ollama",
                extra_args={
                    "context_length": 32768,  # 32K context with Q4
                    "quantization": "Q4_K_M",
                },
            ),
        ],
        min_gpu_ram_gb=24,
        min_cpu_cores=4,
        min_ram_gb=32,
        min_disk_gb=40,
        max_cost_per_hour=0.40,
        idle_pause_minutes=15,
        max_runtime_minutes=180,  # 3 hours for longer sessions
        interruptible=True,
        preferred_gpus=["RTX 4090", "RTX 3090", "A6000", "L40"],
        startup_script=OLLAMA_QWEN3_CODER_STARTUP,
    ),

    # Qwen 7B via Ollama on RTX 4090 (LEGACY - smaller/faster)
    "qwen-ollama": InstanceProfile(
        name="qwen-ollama",
        description="Qwen2.5-Coder-7B via Ollama on RTX 4090 (~160 tok/s) [LEGACY]",
        models=[
            ModelSpec(
                name="qwen-7b",
                model_id="qwen2.5-coder:7b-instruct",
                port=11434,
                vram_gb=8,
                service_type="ollama",
            ),
        ],
        min_gpu_ram_gb=24,
        min_cpu_cores=4,
        min_ram_gb=16,
        min_disk_gb=30,
        max_cost_per_hour=0.35,
        idle_pause_minutes=15,  # Pause after 15 min idle
        max_runtime_minutes=120,  # Destroy after 2 hours
        interruptible=True,
        preferred_gpus=["RTX 4090", "RTX 3090", "A5000", "L40"],
        startup_script=OLLAMA_QWEN_STARTUP,
    ),

    # LLMLingua-2 only (compression)
    "compression": InstanceProfile(
        name="compression",
        description="LLMLingua-2 compression only (cheap, minimal GPU)",
        models=[
            ModelSpec(
                name="llmlingua-2",
                model_id="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                port=8080,
                vram_gb=2,
                service_type="llmlingua",
            ),
        ],
        min_gpu_ram_gb=4,
        min_cpu_cores=2,
        min_ram_gb=4,
        min_disk_gb=10,
        max_cost_per_hour=0.10,
    ),

    # Qwen 7B for code generation
    "qwen-7b": InstanceProfile(
        name="qwen-7b",
        description="Qwen2.5-Coder-7B-Instruct for code generation",
        models=[
            ModelSpec(
                name="qwen-7b",
                model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
                port=8000,
                vram_gb=14,
                service_type="vllm",
                extra_args={
                    "max_model_len": 8192,
                    "gpu_memory_utilization": 0.90,
                },
            ),
        ],
        min_gpu_ram_gb=16,
        min_cpu_cores=4,
        min_ram_gb=16,
        min_disk_gb=30,
        max_cost_per_hour=0.30,
    ),

    # Both models on same GPU (24GB+)
    "full": InstanceProfile(
        name="full",
        description="LLMLingua-2 + Qwen2.5-Coder on same GPU",
        models=[
            ModelSpec(
                name="llmlingua-2",
                model_id="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                port=8080,
                vram_gb=2,
                service_type="llmlingua",
            ),
            ModelSpec(
                name="qwen-7b",
                model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
                port=8000,
                vram_gb=14,
                service_type="vllm",
                extra_args={
                    "max_model_len": 4096,  # Reduced for shared GPU
                    "gpu_memory_utilization": 0.70,
                },
            ),
        ],
        min_gpu_ram_gb=24,
        min_cpu_cores=8,
        min_ram_gb=32,
        min_disk_gb=50,
        max_cost_per_hour=0.50,
    ),

    # Full config with large context (48GB+)
    "full-large": InstanceProfile(
        name="full-large",
        description="LLMLingua-2 + Qwen with full 32K context",
        models=[
            ModelSpec(
                name="llmlingua-2",
                model_id="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                port=8080,
                vram_gb=2,
                service_type="llmlingua",
            ),
            ModelSpec(
                name="qwen-7b",
                model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
                port=8000,
                vram_gb=14,
                service_type="vllm",
                extra_args={
                    "max_model_len": 32768,
                    "gpu_memory_utilization": 0.85,
                },
            ),
        ],
        min_gpu_ram_gb=48,
        min_cpu_cores=8,
        min_ram_gb=64,
        min_disk_gb=100,
        max_cost_per_hour=1.00,
    ),

    # DeepSeek Coder
    "deepseek-7b": InstanceProfile(
        name="deepseek-7b",
        description="DeepSeek-Coder-7B-Instruct",
        models=[
            ModelSpec(
                name="deepseek-7b",
                model_id="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
                port=8000,
                vram_gb=14,
                service_type="vllm",
                extra_args={
                    "max_model_len": 8192,
                    "gpu_memory_utilization": 0.90,
                },
            ),
        ],
        min_gpu_ram_gb=16,
        min_cpu_cores=4,
        min_ram_gb=16,
        min_disk_gb=30,
        max_cost_per_hour=0.30,
    ),
}


def get_profile(name: str) -> InstanceProfile:
    """Get a profile by name."""
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    return PROFILES[name]


def list_profiles() -> list[dict[str, Any]]:
    """List all available profiles."""
    return [
        {
            "name": p.name,
            "description": p.description,
            "models": [m.name for m in p.models],
            "ports": p.get_ports(),
            "min_gpu_ram_gb": p.min_gpu_ram_gb,
            "max_cost_per_hour": p.max_cost_per_hour,
        }
        for p in PROFILES.values()
    ]


def create_custom_profile(
    name: str,
    models: list[ModelSpec],
    description: str = "",
    max_cost_per_hour: float = 1.0,
    **kwargs,
) -> InstanceProfile:
    """Create a custom profile with specified models."""
    return InstanceProfile(
        name=name,
        description=description or f"Custom profile: {name}",
        models=models,
        max_cost_per_hour=max_cost_per_hour,
        **kwargs,
    )
