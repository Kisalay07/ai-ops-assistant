from __future__ import annotations

import json
from typing import Optional

from agents.schemas import AgentPlan, VerificationResult
from llm.groq_client import GroqClient, safe_json_loads
from llm.prompts import VERIFIER_SYSTEM, VERIFIER_USER_TEMPLATE


class VerifierAgent:
    """Verifier Agent: validates results and requests fixes if needed."""

    def __init__(self, llm: Optional[GroqClient] = None) -> None:
        self.llm = llm or GroqClient()

    def verify(self, task: str, plan: AgentPlan, results: dict) -> VerificationResult:
        user_prompt = VERIFIER_USER_TEMPLATE.format(
            task=task,
            plan_json=json.dumps(plan.model_dump(), ensure_ascii=False, indent=2),
            results_json=json.dumps(results, ensure_ascii=False, indent=2),
        )
        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        last_err: Optional[str] = None
        last_text: Optional[str] = None

        for _attempt in range(3):
            resp = self.llm.chat(messages, temperature=0.1, max_tokens=1600, json_mode=True)
            last_text = resp.content
            try:
                data = safe_json_loads(resp.content)
                return VerificationResult.model_validate(data)
            except Exception as e:
                last_err = str(e)
                repair_user = (
                    "Your previous output was invalid JSON or did not match the schema. "
                    "Fix it and output ONLY valid JSON.\n"
                    f"Error: {last_err}\n"
                    "Previous output:\n"
                    f"{last_text}"
                )
                messages = [
                    {"role": "system", "content": VERIFIER_SYSTEM},
                    {"role": "user", "content": repair_user},
                ]

        raise RuntimeError(f"VerifierAgent failed to produce valid verification JSON. Last error: {last_err}")
