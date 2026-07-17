"""Planner node (Task 1.2).

TODO: Implement `make_planner(llm)` returning a node that:
  - reads the user question from state["messages"],
  - asks the LLM (PLANNER_PROMPT) for a JSON list of 2-5 steps,
  - parses it robustly (fallback to a single step on parse failure),
  - returns {"plan": [...], "current_step_index": 0, "step_results": []}.
"""

from __future__ import annotations

from agent.state import AnalystState


      
def _message_content(message: Any) -> str:
    """Return text from either a LangChain message or an input message dict."""
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _parse_plan(content: str) -> list[str] | None:
    """Parse and validate a JSON plan containing two to five nonempty steps."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(parsed, list) or not 2 <= len(parsed) <= 5:
        return None
    if not all(isinstance(step, str) and step.strip() for step in parsed):
        return None
    return [step.strip() for step in parsed]

def make_planner(llm):
    """Create a planner node backed by the supplied chat model."""

    def planner(state: AnalystState) -> dict:
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("Planner requires at least one message")

        user_query = _message_content(messages[-1]).strip()
        if not user_query:
            raise ValueError("Planner received an empty user query")

        response = llm.invoke(
            [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(content=user_query),
            ]
        )
        plan = _parse_plan(_message_content(response))
        if plan is None:
            plan = [user_query]

        return {
            "plan": plan,
            "current_step_index": 0,
            "step_results": [],
        }

    return planner

