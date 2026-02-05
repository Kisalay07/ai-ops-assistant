from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from agents.schemas import ToolResult, ToolName


class BaseTool(ABC):
    name: ToolName

    @abstractmethod
    def call(self, tool_args: Dict[str, Any]) -> ToolResult:
        raise NotImplementedError
