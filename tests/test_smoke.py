import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from langchain_core.messages import AIMessage  # noqa: E402


class FakeDocument:
    page_content = "FY2023 revenue was 100 million dollars."
    metadata = {"source": "annual_report.pdf", "page": 7}


class FakeRetriever:
    def invoke(self, query):
        return [FakeDocument()]


class FakeTool:
    name = "calculate"

    def invoke(self, arguments):
        expression = arguments["expression"]
        return f"{expression} = 110 million dollars"


class FakeLLM:
    """Return deterministic responses for each graph node."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        system_prompt = messages[0].content.lower()

        if "planner" in system_prompt:
            return AIMessage(
                content=(
                    '["Find FY2023 revenue in the document", '
                    '"Using the result from step 1, calculate a 10% increase"]'
                )
            )

        if "routing" in system_prompt or "route one" in system_prompt:
            current_step = messages[-1].content.lower()
            route = "mcp_tools" if "calculate" in current_step else "rag_agent"
            return AIMessage(content=route)

        if "extract" in system_prompt:
            return AIMessage(
                content=(
                    "FY2023 revenue was 100 million dollars "
                    "[source: annual_report.pdf, p.7]"
                )
            )

        if "numerical plan step" in system_prompt:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {"expression": "100 * 1.10"},
                        "id": "smoke-call-1",
                        "type": "tool_call",
                    }
                ],
            )

        if "final-answer synthesizer" in system_prompt:
            return AIMessage(
                content=(
                    "Step 1 found revenue of 100 million dollars "
                    "[source: annual_report.pdf, p.7]. "
                    "Step 2 calculated a 10% increase to 110 million dollars."
                )
            )

        raise AssertionError(f"Unexpected system prompt: {system_prompt}")


def test_graph_module_imports():
    """Minimal collection guard: the graph module must import cleanly."""
    from agent.graph import build_graph  # noqa: F401


def test_full_graph_offline_smoke():
    """The combined query must run through both specialists and synthesize."""
    from agent.graph import build_graph

    graph = build_graph(
        llm=FakeLLM(),
        retriever=FakeRetriever(),
        tools=[FakeTool()],
    )
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What was FY2023 revenue, and what would it be "
                        "after a 10% increase?"
                    ),
                }
            ]
        }
    )

    assert result["plan"]
    assert len(result["step_results"]) == 2
    assert "annual_report.pdf" in result["step_results"][0]
    assert "110 million dollars" in result["step_results"][1]
    assert result["messages"]
    assert result["messages"][-1].content
