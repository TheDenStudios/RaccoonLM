"""RaccoonLM v2 — Plugin System

Plugins extend the model's capabilities through tool calling.
Each plugin inherits from `Plugin` (base.py) and provides:
- Tool definitions for Ollama
- Async tool execution
- Cleanup on shutdown

Usage:
    from raccoonlm.plugins.base import Plugin
    from raccoonlm.plugins.internet import InternetPlugin
"""

from raccoonlm.plugins.base import Plugin
from raccoonlm.plugins.internet import InternetPlugin

__all__ = ["Plugin", "InternetPlugin"]
