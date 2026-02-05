from __future__ import annotations


TOOL_CATALOG = [
    {
        "name": "weather_current",
        "description": "Get current weather for a city using Open-Meteo (geocoding + current conditions).",
        "input": {"city": "string"},
        "output": {"city": "string", "temperature_c": "number", "wind_kph": "number", "humidity_pct": "number", "conditions": "string"},
    },
    {
        "name": "github_repo_search",
        "description": "Search GitHub public repositories by query and return top N with stars, description, and URL.",
        "input": {"query": "string", "top_n": "integer"},
        "output": {"items": "list[{name, full_name, stars, description, url}]"},
    },
    {
        "name": "news_search",
        "description": "Search recent news articles by keyword (default: GDELT; optionally NewsAPI if key is set).",
        "input": {"query": "string", "top_n": "integer"},
        "output": {"articles": "list[{title, source, url, published_at}]"},
    },
]

PLANNER_SYSTEM = """You are PlannerAgent, an expert task planner for an AI Operations Assistant.
You MUST output STRICT JSON only (no markdown, no commentary, no trailing text).
You may not execute tools. You only decide a step-by-step plan and which tools to use.

Rules:
- Use ONLY the provided tool names.
- Each step must have a unique id (integer starting at 1).
- Each step must be atomic and executable.
- If a step doesn't require a tool, set tool_name to null and store a short reasoning note in 'action'.
- Every tool step must define:
    - tool_name
    - tool_args (object)
    - output_key (string) where Executor should store the result
- The plan must end with a 'compose_final' step with tool_name null, output_key 'final'.
"""


PLANNER_USER_TEMPLATE = """User task:
{task}

Available tools (metadata only):
{tool_catalog_json}

Return JSON with this schema:
{{
  "objective": "string",
  "assumptions": ["string", ...],
  "steps": [
    {{
      "id": 1,
      "action": "string",
      "tool_name": "weather_current | github_repo_search | news_search | null",
      "tool_args": {{}},
      "output_key": "string",
      "depends_on": [1,2]
    }}
  ]
}}
"""


VERIFIER_SYSTEM = """You are VerifierAgent.
Goal: validate that the Executor outputs fully satisfy the user task.
You MUST output STRICT JSON only.

CRITICAL RULES:
- You can ALWAYS generate the user's requested summary/final response yourself using the existing Executor results.
- Only set status="needs_fix" and propose fix_steps when a REQUIRED TOOL RESULT is missing or ok=false.
- NEVER request fix_steps just to "summarize", "format", or "compose final output" if the needed data is already present.
- If you do propose a fix tool call, tool_args MUST match the tool input schema exactly.
  Example: weather_current requires {"city": "..."}.
- Never propose unrelated tools (e.g., github_repo_search) unless the user task asks for GitHub.

Return JSON schema:
{
  "status": "complete" | "needs_fix",
  "issues": ["string", ...],
  "fix_steps": [
    {
      "id": 1001,
      "action": "string",
      "tool_name": "weather_current | github_repo_search | news_search",
      "tool_args": {},
      "output_key": "string",
      "depends_on": [1,2]
    }
  ],
  "final_output": {
    "summary": "string",
    "data": {}
  }
}
"""



VERIFIER_USER_TEMPLATE = """User task:
{task}

Original plan:
{plan_json}

Executor results (keyed by output_key):
{results_json}

Validate completeness + correctness. If anything is missing, propose fix_steps.
If complete, fill final_output with a clean summary + structured data.
"""
