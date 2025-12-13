"""Cloud GPU instance management for Dumont Core.

Simple interface for GPU instances with LLM services via Skypilot/Vast.ai.

=== Quick Start ===

    from dumont_core.cloud import quick

    # Check status
    quick.status()

    # Start GPU (creates if needed)
    await quick.start()

    # Test the model
    response = await quick.test("Write hello world in Python")

    # Stop when done
    await quick.stop()

=== Configuration ===

    Default model: qwen3-coder:30b-a3b-q4_K_M
    Max cost: $0.40/hr
    Auto-pause: 15 min idle
    Max runtime: 3 hours

    Config file: ~/.lca/gpu_config.json
    State file: ~/.lca/gpu_state.json
"""

# Core modules
from .config import (
    CloudConfig,
    get_config,
    set_model,
    list_available_models,
    DEFAULT_MODEL,
    MODELS,
)
from .profiles import (
    ModelSpec,
    InstanceProfile,
    PROFILES,
    get_profile,
    list_profiles,
    create_custom_profile,
)
from .gpu_lifecycle import (
    GPUInstanceState,
)
from . import quick

__all__ = [
    # Config
    "CloudConfig",
    "get_config",
    "set_model",
    "list_available_models",
    "DEFAULT_MODEL",
    "MODELS",
    # Profiles
    "ModelSpec",
    "InstanceProfile",
    "PROFILES",
    "get_profile",
    "list_profiles",
    "create_custom_profile",
    # GPU State
    "GPUInstanceState",
    # Quick functions
    "quick",
]
