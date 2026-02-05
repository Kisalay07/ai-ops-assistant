from __future__ import annotations

import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import typer

from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from agents.verifier import VerifierAgent
from llm.groq_client import GroqClient
from utils.logging import setup_logging


load_dotenv(override=True)
setup_logging()

app = FastAPI(
    title="AI Operations Assistant",
    version="1.0.0",
    description="Multi-agent AI Ops Assistant .",
)

class RunRequest(BaseModel):
    task: str = Field(..., min_length=1)
    max_rounds: int = Field(2, ge=1, le=5)

class RunResponse(BaseModel):
    task: str
    plan: Dict[str, Any]
    results: Dict[str, Any]
    verification: Dict[str, Any]
    final_output: Dict[str, Any]
    logs: list[str]

def _build_agents() -> tuple[PlannerAgent, ExecutorAgent, VerifierAgent]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Set it in .env or environment variables.")

    llm = GroqClient(api_key=api_key)
    return PlannerAgent(llm=llm), ExecutorAgent(llm=llm), VerifierAgent(llm=llm)


def run_task(task: str, max_rounds: int = 2) -> RunResponse:
    planner, executor, verifier = _build_agents()

    plan = planner.plan(task)
    state = executor.run(task, plan)

    verification = verifier.verify(task, plan, state.results)

    rounds = 1
    while verification.status == "needs_fix" and verification.fix_steps and rounds < max_rounds:
        state = executor.run_fix_steps(state, verification.fix_steps)
        verification = verifier.verify(task, plan, state.results)
        rounds += 1

    return RunResponse(
        task=task,
        plan=plan.model_dump(),
        results=state.results,
        verification=verification.model_dump(),
        final_output=verification.final_output,
        logs=state.logs,
    )

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    try:
        return run_task(req.task, max_rounds=req.max_rounds)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- CLI ----
cli = typer.Typer(add_completion=False, help="AI Operations Assistant (CLI).")

@cli.command()
def run_cli(task: str = typer.Argument(..., help="Natural language task to execute"),
            rounds: int = typer.Option(2, help="Max verify/fix rounds")):
    """Run a task from the command line."""
    res = run_task(task, max_rounds=rounds)
    # Print final output nicely
    import json
    typer.echo(json.dumps(res.final_output, ensure_ascii=False, indent=2))

@cli.command()
def plan(task: str = typer.Argument(..., help="Task to plan (Planner only)")):
    """Generate and print the Planner JSON plan (no execution)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise typer.BadParameter("GROQ_API_KEY is missing. Set it in .env or environment variables.")
    llm = GroqClient(api_key=api_key)
    planner = PlannerAgent(llm=llm)
    plan_obj = planner.plan(task)
    import json
    typer.echo(json.dumps(plan_obj.model_dump(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    cli()
