"""
Backwards compatibility module for dumont_core imports.

The actual modules are in the root of dumont-core.
This re-exports them for compatibility with existing code.
"""

import sys
import os

# Get the parent directory (dumont-core root)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Re-export modules from root
from llm import *
from llm import get_llm_manager, get_dedicated_provider, LLMProvider, DedicatedBackend

# Re-export testing module
from . import testing
from . import llm as llm_module
from . import cloud as cloud_module

__all__ = ['llm', 'testing', 'cloud', 'get_llm_manager', 'get_dedicated_provider', 'LLMProvider', 'DedicatedBackend']
