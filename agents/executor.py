from __future__ import annotations

import json
from typing import List, Optional

from agents.schemas import AgentPlan, ExecutionState, PlanStep, ToolResult, VerifierStep
from llm.groq_client import GroqClient
from tools.registry import get_tool
from utils.logging import get_logger

logger = get_logger(__name__)


class ExecutorAgent:
    """Executor Agent: executes plan steps sequentially and calls tools."""

    def __init__(self, llm: Optional[GroqClient] = None) -> None:
        self.llm = llm

    def run(self, task: str, plan: AgentPlan) -> ExecutionState:
        state = ExecutionState(task=task, plan=plan)
        self._run_steps(plan.steps, state)
        return state

    def run_fix_steps(self, state: ExecutionState, fix_steps: List[VerifierStep]) -> ExecutionState:
        pseudo_steps: List[PlanStep] = []
        for fs in fix_steps:
            pseudo_steps.append(
                PlanStep(
                    id=fs.id,
                    action=fs.action,
                    tool_name=fs.tool_name,
                    tool_args=fs.tool_args,
                    output_key=fs.output_key,
                    depends_on=fs.depends_on,
                )
            )
        self._run_steps(pseudo_steps, state)
        return state

    def _deps_ok(self, step: PlanStep, state: ExecutionState) -> bool:
        for dep in step.depends_on:
            if state.step_status.get(dep) != "ok":
                return False
        return True

    def _compose_text(self, state: ExecutionState, step: PlanStep) -> str:
        # If no LLM provided, fall back to a simple note.
        if not self.llm:
            return step.action

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Executor's composer. You MUST NOT call tools. "
                    "Use ONLY the provided tool results to complete the step. "
                    "If asked for bullet points, output bullet points. Output only the text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task: {state.task}\n"
                    f"Step action: {step.action}\n\n"
                    f"Available results JSON:\n{json.dumps(state.results, ensure_ascii=False, indent=2)}"
                ),
            },
        ]
        resp = self.llm.chat(messages, temperature=0.2, max_tokens=500, json_mode=False)
        return resp.content.strip()

    def _run_steps(self, steps: List[PlanStep], state: ExecutionState) -> None:
        for step in steps:
            if not self._deps_ok(step, state):
                state.step_status[step.id] = "skipped"
                state.logs.append(f"Step {step.id} skipped due to failed dependency: {step.depends_on}")
                continue

            if step.tool_name is None:
                text = self._compose_text(state, step)
                state.step_status[step.id] = "ok"
                state.results[step.output_key] = {"text": text}
                state.logs.append(f"Step {step.id} composed text under '{step.output_key}'")
                continue

            tool = get_tool(step.tool_name)
            state.logs.append(f"Step {step.id} calling tool '{step.tool_name}' with args={step.tool_args}")

            try:
                tool_res: ToolResult = tool.call(step.tool_args)
            except Exception as e:
                tool_res = ToolResult(ok=False, tool_name=step.tool_name, error=str(e))

            state.results[step.output_key] = tool_res.model_dump()
            state.step_status[step.id] = "ok" if tool_res.ok else "failed"

            if tool_res.ok:
                state.logs.append(f"Step {step.id} ok -> stored '{step.output_key}'")
            else:
                state.logs.append(f"Step {step.id} failed: {tool_res.error}")
