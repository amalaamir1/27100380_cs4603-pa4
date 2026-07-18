"""Full Document Analyst graph (Tasks 1.5 + 1.7).

TODO:
  - `load_mcp_tools(server_path=None)`: connect the GIVEN MCP server over stdio
    (see langchain-mcp-adapters) and return its tools.
  - `make_mcp_node(tools, llm)`: execute one calculation step by letting the LLM
    call exactly one MCP tool, then append the result and increment the index.
  - `build_graph(llm=None, retriever=None, tools=None)`: assemble
    planner -> supervisor -> {rag_agent | mcp_tools} -> ... -> synthesizer.
    Inject dependencies so the graph can be unit-tested offline with fakes.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer



def load_mcp_tools(server_path: str | None = None):
    """Launch the bundled MCP server over stdio and load its tools once."""
    if server_path is None:
        server = Path(__file__).resolve().parents[1] / "tools" / "mcp_server.py"
    else:
        server = Path(server_path).expanduser().resolve()

    if not server.is_file():
        raise FileNotFoundError(f"MCP server not found: {server}")

    client = MultiServerMCPClient(
        {
            "analyst": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(server)],
            }
        }
    )
    # Graph construction is synchronous. Loading here ensures tool discovery occurs
    # once rather than starting a discovery session on every graph step.
    return asyncio.run(client.get_tools())


def _message_text(message) -> str:
    """Normalize an LLM or tool response to text for step_results."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def make_mcp_node(tools, llm):
    """Create a synchronous node that executes exactly one MCP calculation tool."""
    if not tools:
        raise ValueError("At least one MCP tool is required")

    tools_by_name = {tool.name: tool for tool in tools}
    tool_llm = llm.bind_tools(tools)

    def mcp_tools(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        step_index = state.get("current_step_index", 0)
        if step_index >= len(plan):
            raise ValueError("MCP node requires an unfinished plan step")

        current_step = plan[step_index]
        previous_results = state.get("step_results", [])
        prior_context = "\n".join(
            f"Step {index + 1}: {result}"
            for index, result in enumerate(previous_results)
        ) or "None"

        decision = tool_llm.invoke(
            [
                SystemMessage(content=MCP_STEP_PROMPT),
                HumanMessage(
                    content=(
                        f"Current calculation step:\n{current_step}\n\n"
                        f"Results from earlier steps:\n{prior_context}"
                    )
                ),
            ]
        )
        tool_calls = getattr(decision, "tool_calls", [])
        if len(tool_calls) != 1:
            raise ValueError(
                f"MCP step must produce exactly one tool call; received {len(tool_calls)}"
            )

        call = tool_calls[0]
        tool_name = call.get("name")
        if tool_name not in tools_by_name:
            raise ValueError(f"LLM selected an unknown MCP tool: {tool_name!r}")

        # Synchronous invocation is intentional for the bundled serving-container MCP.
        tool_result = tools_by_name[tool_name].invoke(call.get("args", {}))
        result_text = _message_text(tool_result)
        if not result_text:
            result_text = "MCP tool returned no result"

        return {
            "step_results": [*previous_results, result_text],
            "current_step_index": step_index + 1,
        }

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    """Task 1.7: full graph wiring is intentionally implemented later."""
 
    builder = StateGraph(AnalystState)
    builder.add_node("planner", planner)
    builder.add_node("supervisor", supervisor)
    builder.add_node("rag_agent", rag_agent)
    builder.add_node("mcp_tools", mcp_tools)
    builder.add_node("synthesizer", synthesizer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges("supervisor", route_from_supervisor)
    builder.add_edge("rag_agent", "supervisor")
    builder.add_edge("mcp_tools", "supervisor")
    builder.add_edge("synthesizer", END)

    graph = builder.compile()