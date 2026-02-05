from __future__ import annotations

from typing import Dict

from tools.github_tool import GitHubTool
from tools.weather_tool import WeatherTool
from tools.news_tool import NewsTool
from tools.base import BaseTool


_TOOL_REGISTRY: Dict[str, BaseTool] = {
    "github_repo_search": GitHubTool(),
    "weather_current": WeatherTool(),
    "news_search": NewsTool(),
}

def get_tool(tool_name: str) -> BaseTool:
    if tool_name not in _TOOL_REGISTRY:
        raise KeyError(f"Unknown tool: {tool_name}")
    return _TOOL_REGISTRY[tool_name]
