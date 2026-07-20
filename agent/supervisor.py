from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import SUPERVISOR_PROMPT
from agent.state import AnalystState

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"


_CALCULATION_KEYWORDS = {
    "calculate",
    "compute",
    "percentage",
    "percent",
    "growth",
    "cagr",
    "increase",
    "decrease",
    "difference",
    "compare",
    "convert",
    "ratio",
    "multiply",
    "divide",
    "sum",
    "average",
}


def _route_from_response(content: str, current_step: str) -> str:
    """Normalize the model response, with a deterministic fallback."""
    decision = content.strip().lower().replace("-", "_").replace(" ", "_")
    if decision == RAG or "rag_agent" in decision or decision == "rag":
        return RAG
    if decision == MCP or "mcp_tools" in decision or decision in {"mcp", "math"}:
        return MCP

    words = set(current_step.lower().replace("%", " percent ").split())
    return MCP if words & _CALCULATION_KEYWORDS else RAG

def make_supervisor(llm):
    
    def supervisor(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        current_step_index = state.get("current_step_index", 0)

        if current_step_index >= len(plan):
            return {"next_agent": SYNTH}

        current_step = plan[current_step_index]
        response = llm.invoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(content=current_step),
            ]
        )
        content = getattr(response, "content", "")
        route = _route_from_response(str(content), current_step)
        return {"next_agent": route}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    """Return the node selected by the supervisor for a conditional edge."""
    route = state["next_agent"]
    if route not in {RAG, MCP, SYNTH}:
        raise ValueError(f"Unknown supervisor route: {route!r}")
    return route