from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import SYNTHESIZER_PROMPT
from agent.state import AnalystState


def _build_step_context(state: AnalystState) -> str:
    """Pair each completed result with the plan step that produced it."""
    plan = state.get("plan", [])
    results = state.get("step_results", [])
    if not results:
        return "No steps produced a result."

    entries = []
    for index, result in enumerate(results):
        step = plan[index] if index < len(plan) else "Unspecified step"
        entries.append(
            f"Step {index + 1}\n"
            f"Task: {step}\n"
            f"Result: {result}"
        )
    return "\n\n".join(entries)


def _response_text(response) -> str:
    content = getattr(response, "content", "")
    return content.strip() if isinstance(content, str) else str(content).strip()


def make_synthesizer(llm):
    """Return a node that converts all step results into the final chat answer."""

    def synthesizer(state: AnalystState) -> dict:
        context = _build_step_context(state)
        response = llm.invoke(
            [
                SystemMessage(content=SYNTHESIZER_PROMPT),
                HumanMessage(content=f"Completed plan steps and results:\n\n{context}"),
            ]
        )
        final_answer = _response_text(response)
        if not final_answer:
            final_answer = "I could not produce a final answer from the completed steps."

        return {
            "final_answer": final_answer,
            "messages": [AIMessage(content=final_answer)],
        }

    return synthesizer
