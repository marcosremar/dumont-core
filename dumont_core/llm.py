"""Backwards compatibility - re-export llm module from root."""
import sys
import os

# Add parent to path
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# Re-export everything from root llm module
from llm import *
from llm import get_llm_manager, get_dedicated_provider, LLMProvider, DedicatedBackend
