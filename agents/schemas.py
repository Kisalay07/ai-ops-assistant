from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator


ToolName = Literal["weather_current", "github_repo_search", "news_search"]

class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(..., ge=1)
    action: str = Field(..., min_length=1)
    tool_name: Optional[ToolName] = None
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    output_key: str = Field(..., min_length=1)
    depends_on: List[int] = Field(default_factory=list)

    @field_validator("depends_on")
    @classmethod
    def depends_on_no_self(cls, v: List[int], info):
        step_id = info.data.get("id")
        if step_id and step_id in v:
            raise ValueError("depends_on cannot include step id itself")
        return v

class AgentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field(..., min_length=1)
    assumptions: List[str] = Field(default_factory=list)
    steps: List[PlanStep] = Field(..., min_length=1)

    @field_validator("steps")
    @classmethod
    def unique_ids(cls, steps: List[PlanStep]):
        ids = [s.id for s in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Step ids must be unique")
        # Ensure there's a compose_final at the end
        if steps[-1].output_key != "final" or steps[-1].tool_name is not None:
            raise ValueError("Last step must be compose_final (tool_name=null, output_key='final')")
        return steps

class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    tool_name: ToolName
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

class ExecutionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    plan: AgentPlan
    results: Dict[str, Any] = Field(default_factory=dict)  # output_key -> tool result / notes
    step_status: Dict[int, str] = Field(default_factory=dict)  # id -> ok/failed/skipped
    logs: List[str] = Field(default_factory=list)

class VerifierStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(..., ge=1000)
    action: str
    tool_name: ToolName
    tool_args: Dict[str, Any]
    output_key: str
    depends_on: List[int] = Field(default_factory=list)

class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["complete", "needs_fix"]
    issues: List[str] = Field(default_factory=list)
    fix_steps: List[VerifierStep] = Field(default_factory=list)
    final_output: Dict[str, Any] = Field(default_factory=dict)
