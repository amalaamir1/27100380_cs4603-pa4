from __future__ import annotations
from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import RAG_EXTRACT_PROMPT
from agent.state import AnalystState
NOT_FOUND = "not found in documents"



def format_docs(docs) -> str:
    """Format retrieved chunks with the citations required by the assignment."""
    formatted = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source", "unknown source")
        page = metadata.get("page", "unknown")
        citation = f"[source: {source}, p.{page}]"
        content = str(getattr(doc, "page_content", "")).strip()
        formatted.append(f"{content}\n{citation}")
    return "\n\n".join(formatted)


def _response_text(response) -> str:
    content = getattr(response, "content", "")
    return content.strip() if isinstance(content, str) else str(content).strip()


def make_rag_agent(retriever, llm):
    """Return a node that retrieves and extracts one cited fact for a plan step."""

    def rag_agent(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        step_index = state.get("current_step_index", 0)
        if step_index >= len(plan):
            raise ValueError("RAG agent requires an unfinished plan step")

        current_step = plan[step_index]
        docs = retriever.invoke(current_step)

        if not docs:
            fact = NOT_FOUND
        else:
            context = format_docs(docs)
            response = llm.invoke(
                [
                    SystemMessage(content=RAG_EXTRACT_PROMPT),
                    HumanMessage(
                        content=f"Current step:\n{current_step}\n\nRetrieved chunks:\n{context}"
                    ),
                ]
            )
            fact = _response_text(response) or NOT_FOUND

        return {
            "step_results": [*state.get("step_results", []), fact],
            "current_step_index": step_index + 1,
        }

    return rag_agent
