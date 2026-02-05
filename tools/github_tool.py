from __future__ import annotations

from typing import Any, Dict, List

import requests

from agents.schemas import ToolResult
from tools.base import BaseTool
from utils.retry import with_retry, RetryableError


class GitHubTool(BaseTool):
    """GitHub public repository search ."""

    name = "github_repo_search"

    def __init__(self, timeout_s: float = 20.0) -> None:
        self.timeout_s = timeout_s
        self.base_url = "https://api.github.com"

    @with_retry(attempts=3)
    def _get(self, path: str, params: Dict[str, Any]) -> requests.Response:
        url = f"{self.base_url}{path}"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "ai-ops-assistant",
        }
        try:
            r = requests.get(url, params=params, headers=headers, timeout=self.timeout_s)
        except requests.RequestException as e:
            raise RetryableError(f"GitHub request failed: {e}") from e

        if r.status_code in (429, 500, 502, 503, 504):
            raise RetryableError(f"GitHub transient error {r.status_code}: {r.text[:200]}")

        # GitHub rate limit returns 403 with message
        if r.status_code == 403 and "rate limit" in r.text.lower():
            raise RetryableError(f"GitHub rate limited: {r.text[:200]}")

        return r

    def call(self, tool_args: Dict[str, Any]) -> ToolResult:
        query = str(tool_args.get("query", "")).strip()
        top_n = int(tool_args.get("top_n", 5))

        if not query:
            return ToolResult(ok=False, tool_name=self.name, error="Missing 'query'")

        top_n = max(1, min(top_n, 20))
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": top_n,
        }

        try:
            r = self._get("/search/repositories", params=params)
        except Exception as e:
            return ToolResult(ok=False, tool_name=self.name, error=str(e))

        if not r.ok:
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error=f"GitHub error {r.status_code}: {r.text[:200]}",
                meta={"status_code": r.status_code},
            )

        data = r.json()
        items: List[Dict[str, Any]] = []
        for it in data.get("items", [])[:top_n]:
            items.append(
                {
                    "name": it.get("name"),
                    "full_name": it.get("full_name"),
                    "stars": it.get("stargazers_count"),
                    "description": it.get("description"),
                    "url": it.get("html_url"),
                    "language": it.get("language"),
                }
            )

        return ToolResult(
            ok=True,
            tool_name=self.name,
            data={"query": query, "items": items},
            meta={"total_count": data.get("total_count"), "status_code": r.status_code},
        )
