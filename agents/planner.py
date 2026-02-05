from __future__ import annotations

import json
from typing import Optional

from agents.schemas import AgentPlan
from llm.groq_client import GroqClient, safe_json_loads
from llm.prompts import PLANNER_SYSTEM, PLANNER_USER_TEMPLATE, TOOL_CATALOG


class PlannerAgent:
    """Planner Agent: turns user task into an executable JSON plan (no tool execution)."""

    def __init__(self, llm: Optional[GroqClient] = None) -> None:
        self.llm = llm or GroqClient()

    def plan(self, task: str) -> AgentPlan:
        tool_catalog_json = json.dumps(TOOL_CATALOG, ensure_ascii=False, indent=2)
        user_prompt = PLANNER_USER_TEMPLATE.format(
            task=task,
            tool_catalog_json=tool_catalog_json,
        )
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        last_err: Optional[str] = None
        last_text: Optional[str] = None

        for _attempt in range(3):
            resp = self.llm.chat(messages, temperature=0.1, max_tokens=1400, json_mode=True)
            last_text = resp.content
            try:
                data = safe_json_loads(resp.content)
                return AgentPlan.model_validate(data)
            except Exception as e:
                last_err = str(e)
                # Ask model to repair output (strict JSON only)
                repair_user = (
                    "Your previous output was invalid JSON or did not match the schema. "
                    "Fix it and output ONLY valid JSON.\n"
                    f"Error: {last_err}\n"
                    "Previous output:\n"
                    f"{last_text}"
                )
                messages = [
                    {"role": "system", "content": PLANNER_SYSTEM},
                    {"role": "user", "content": repair_user},
                ]

        raise RuntimeError(f"PlannerAgent failed to produce valid plan. Last error: {last_err}")
