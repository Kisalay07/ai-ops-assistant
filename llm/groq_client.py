from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from utils.retry import with_retry, RetryableError

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

@dataclass(frozen=True)
class LLMResponse:
    content: str
    raw: Dict[str, Any]

class GroqClient:
    """Minimal OpenAI-compatible chat client for Groq."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        self.timeout_s = timeout_s

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your environment or .env file."
            )

    @with_retry(attempts=4)
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1200,
        json_mode: bool = False,
    ) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

       
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            r = requests.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            raise RetryableError(f"Groq request failed: {e}") from e

        if r.status_code in (429, 500, 502, 503, 504):
            raise RetryableError(f"Groq transient error {r.status_code}: {r.text[:200]}")

        if not r.ok:
            raise RuntimeError(f"Groq error {r.status_code}: {r.text}")

        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return LLMResponse(content=content, raw=data)

def safe_json_loads(text: str) -> Any:
    """Parse JSON robustly; raises ValueError if impossible."""
    text = text.strip()

    
    try:
        return json.loads(text)
    except Exception:
        pass

    
    import re

    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if match:
        candidate = match.group(1)
        return json.loads(candidate)

    raise ValueError("Could not parse JSON from model output.")
