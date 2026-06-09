"""RaccoonLM v2 — Plugin base class

All plugins must inherit from `Plugin` and implement:
- name: str (unique plugin identifier)
- get_tool_definitions() -> list[dict] (Ollama-compatible tool definitions)
- execute_tool(name, args) -> str (tool execution)
- shutdown() (cleanup on server shutdown)
"""

from abc import ABC, abstractmethod
from typing import Any


class Plugin(ABC):
    """Abstract base class for RaccoonLM plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique plugin name (e.g., 'internet', 'code', 'weather')."""
        ...

    @abstractmethod
    def get_tool_definitions(self) -> list[dict]:
        """
        Return Ollama-compatible tool definitions.

        Each tool definition follows the Ollama tools schema:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": { ... }
            }
        }
        """
        ...

    @abstractmethod
    async def execute_tool(self, name: str, args: dict[str, Any]) -> str:
        """
        Execute a tool and return its result as a string.

        Args:
            name: Tool function name (matches a definition from get_tool_definitions)
            args: Arguments dict matching the tool's parameter schema

        Returns:
            String result (JSON, text, or error), will be inserted as tool response
        """
        ...

    async def shutdown(self):
        """Cleanup resources. Override if plugin holds connections."""
        pass

    def __repr__(self) -> str:
        return f"<Plugin:{self.name}>"
