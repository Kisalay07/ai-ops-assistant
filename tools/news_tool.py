from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv, find_dotenv
import requests

from agents.schemas import ToolResult
from tools.base import BaseTool
from utils.retry import RetryableError, with_retry
from dotenv import load_dotenv
load_dotenv()  


class NewsTool(BaseTool):
    """News search tool.

    Priority order:
      1) NewsAPI.org (if NEWSAPI_KEY is set)
      2) GDELT 2.1 Doc API
      3) Google News RSS (no key; best-effort fallback)

    Design goals:
      - Never crash on non-JSON responses (safe parsing).
      - Provide useful error/meta details for debugging.
      - Retry transient failures (429/5xx) and empty bodies.
    """

    name = "news_search"

    

    def __init__(self, timeout_s: float = 25.0) -> None:
        load_dotenv(find_dotenv())
        self.timeout_s = timeout_s
        self.newsapi_key = os.getenv("NEWSAPI_KEY") or ""

    

    @with_retry(attempts=3)
    def _get(
        self,
        url: str,
        params: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        *,
        require_body: bool = True,
    ) -> requests.Response:
        default_headers = {
            "User-Agent": "ai_ops_assistant/1.0",
            "Accept": "application/json, text/plain;q=0.9, */*;q=0.8",
        }
        merged_headers = {**default_headers, **(headers or {})}

        try:
            r = requests.get(url, params=params, headers=merged_headers, timeout=self.timeout_s)
        except requests.RequestException as e:
            raise RetryableError(f"News request failed: {e}") from e

        # Retry transient/rate-limit errors
        if r.status_code in (429, 500, 502, 503, 504):
            raise RetryableError(f"News transient error {r.status_code}: {(r.text or '')[:200]}")

        # Some upstream/proxies return 200 with empty body; retry
        if require_body and not (r.text or "").strip():
            raise RetryableError("News provider returned an empty response body")

        return r

    def _safe_json(self, r: requests.Response) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Return (data, error_message). Never raises JSONDecodeError."""
        try:
            return r.json(), None
        except ValueError:
            ct = r.headers.get("Content-Type", "")
            body_head = (r.text or "").strip()[:300]
            return None, f"status={r.status_code}, content_type={ct}, body_head={body_head!r}"

    # ---------------------------
    # Providers
    # ---------------------------

    def _newsapi_search(self, query: str, top_n: int) -> ToolResult:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "pageSize": top_n,
            "sortBy": "publishedAt",
            "language": "en",
        }
        headers = {"X-Api-Key": self.newsapi_key}

        try:
            r = self._get(url, params=params, headers=headers)
        except Exception as e:
            return ToolResult(ok=False, tool_name=self.name, error=str(e), meta={"provider": "newsapi"})

        if not r.ok:
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error=f"NewsAPI error {r.status_code}: {(r.text or '')[:200]}",
                meta={
                    "provider": "newsapi",
                    "status_code": r.status_code,
                    "content_type": r.headers.get("Content-Type", ""),
                    "final_url": r.url,
                },
            )

        data, parse_err = self._safe_json(r)
        if parse_err:
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error=f"NewsAPI parse error: {parse_err}",
                meta={
                    "provider": "newsapi",
                    "status_code": r.status_code,
                    "content_type": r.headers.get("Content-Type", ""),
                    "final_url": r.url,
                },
            )

        # NewsAPI can return HTTP 200 with an error payload
        if (data or {}).get("status") != "ok":
            msg = (data or {}).get("message") or "Unknown NewsAPI error"
            code = (data or {}).get("code")
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error=f"NewsAPI returned error: {(code or '').strip()} {msg}".strip(),
                meta={
                    "provider": "newsapi",
                    "status_code": r.status_code,
                    "newsapi_code": code,
                    "final_url": r.url,
                },
            )

        articles_in = (data or {}).get("articles") or []
        articles: List[Dict[str, Any]] = []
        for a in articles_in[:top_n]:
            source = (a.get("source") or {}).get("name")
            articles.append(
                {
                    "title": a.get("title"),
                    "source": source,
                    "url": a.get("url"),
                    "published_at": a.get("publishedAt"),
                }
            )

        return ToolResult(
            ok=True,
            tool_name=self.name,
            data={"query": query, "articles": articles, "provider": "newsapi"},
            meta={"provider": "newsapi", "status_code": r.status_code},
        )

    def _gdelt_search(self, query: str, top_n: int) -> ToolResult:
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": top_n,
            "sort": "HybridRel",
        }

        try:
            r = self._get(url, params=params)
        except Exception as e:
            return ToolResult(ok=False, tool_name=self.name, error=str(e), meta={"provider": "gdelt"})

        if not r.ok:
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error=f"GDELT error {r.status_code}: {(r.text or '')[:200]}",
                meta={
                    "provider": "gdelt",
                    "status_code": r.status_code,
                    "content_type": r.headers.get("Content-Type", ""),
                    "final_url": r.url,
                },
            )

        data, parse_err = self._safe_json(r)
        if parse_err:
            hint = ""
            # common: short queries like "AI" can trigger plain text errors
            if len(query) < 3 or "too short" in (r.text or "").lower():
                hint = " Hint: try a longer query like 'artificial intelligence' or 'generative ai'."
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error=f"GDELT parse error: {parse_err}.{hint}",
                meta={
                    "provider": "gdelt",
                    "status_code": r.status_code,
                    "content_type": r.headers.get("Content-Type", ""),
                    "final_url": r.url,
                },
            )

        articles_in = (data or {}).get("articles") or []
        articles: List[Dict[str, Any]] = []
        for a in articles_in[:top_n]:
            articles.append(
                {
                    "title": a.get("title"),
                    "source": a.get("sourceCountry") or a.get("source") or a.get("domain"),
                    "url": a.get("url"),
                    "published_at": a.get("seendate") or a.get("published") or a.get("datetime"),
                }
            )

        return ToolResult(
            ok=True,
            tool_name=self.name,
            data={"query": query, "articles": articles, "provider": "gdelt"},
            meta={"provider": "gdelt", "status_code": r.status_code},
        )

    def _googlenews_rss_search(self, query: str, top_n: int) -> ToolResult:
        # No key required; last-resort fallback.
        url = "https://news.google.com/rss/search"
        params = {
            "q": query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        }
        headers = {"Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8"}

        try:
            r = self._get(url, params=params, headers=headers)
        except Exception as e:
            return ToolResult(ok=False, tool_name=self.name, error=str(e), meta={"provider": "googlenews_rss"})

        if not r.ok:
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error=f"Google News RSS error {r.status_code}: {(r.text or '')[:200]}",
                meta={
                    "provider": "googlenews_rss",
                    "status_code": r.status_code,
                    "content_type": r.headers.get("Content-Type", ""),
                    "final_url": r.url,
                },
            )

        try:
            root = ET.fromstring(r.text)
        except ET.ParseError as e:
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error=f"Google News RSS parse error: {e}",
                meta={
                    "provider": "googlenews_rss",
                    "status_code": r.status_code,
                    "content_type": r.headers.get("Content-Type", ""),
                    "final_url": r.url,
                },
            )

        items = root.findall(".//item")
        articles: List[Dict[str, Any]] = []
        for it in items[:top_n]:
            title = (it.findtext("title") or "").strip()
            link = (it.findtext("link") or "").strip()
            published_at = (it.findtext("pubDate") or "").strip()
            src = it.find("source")
            source = (src.text or "").strip() if src is not None and src.text else None

            articles.append(
                {
                    "title": title or None,
                    "source": source,
                    "url": link or None,
                    "published_at": published_at or None,
                }
            )

        return ToolResult(
            ok=True,
            tool_name=self.name,
            data={"query": query, "articles": articles, "provider": "googlenews_rss"},
            meta={"provider": "googlenews_rss", "status_code": r.status_code},
        )

    # ---------------------------
    # Public tool entry
    # ---------------------------

    def call(self, tool_args: Dict[str, Any]) -> ToolResult:
        query = str(tool_args.get("query", "")).strip()
        top_n = int(tool_args.get("top_n", 5))

        if not query:
            return ToolResult(ok=False, tool_name=self.name, error="Missing 'query'")

        top_n = max(1, min(top_n, 20))

        # 1) Primary: NewsAPI (if key exists)
        if self.newsapi_key:
            res = self._newsapi_search(query, top_n)
            if res.ok:
                return res

            # 2) Fallback: GDELT
            gd = self._gdelt_search(query, top_n)
            if gd.ok:
                gd.meta["fallback_from"] = "newsapi"
                gd.meta["newsapi_error"] = res.error
                return gd

            # 3) Last fallback: Google News RSS
            rss = self._googlenews_rss_search(query, top_n)
            if rss.ok:
                rss.meta["fallback_from"] = "gdelt"
                rss.meta["newsapi_error"] = res.error
                rss.meta["gdelt_error"] = gd.error
                return rss

            # Return the most informative failure (rss has good context too)
            rss.meta["newsapi_error"] = res.error
            rss.meta["gdelt_error"] = gd.error
            return rss

        # No NewsAPI key: start at GDELT, then RSS
        gd = self._gdelt_search(query, top_n)
        if gd.ok:
            return gd

        rss = self._googlenews_rss_search(query, top_n)
        if rss.ok:
            rss.meta["fallback_from"] = "gdelt"
            rss.meta["gdelt_error"] = gd.error
            return rss

        rss.meta["gdelt_error"] = gd.error
        return rss
