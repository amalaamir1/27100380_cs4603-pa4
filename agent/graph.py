from __future__ import annotations

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.prompts import MCP_STEP_PROMPT
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer
from config import get_chat_llm
from rag.store import get_retriever

def _run_async_synchronously(coro):
    """Run an async operation from regular Python, including inside notebooks."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # A notebook event loop is already active, so use a separate thread.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()
    
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
    return _run_async_synchronously(client.get_tools())


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
        tool = tools_by_name[tool_name]
        tool_args = call.get("args", {})

        if hasattr(tool, "ainvoke"):
            tool_result = _run_async_synchronously(
                tool.ainvoke(tool_args)
            )
        else:
            tool_result = tool.invoke(tool_args)
        result_text = _message_text(tool_result)
        if not result_text:
            result_text = "MCP tool returned no result"

        return {
            "step_results": [*previous_results, result_text],
            "current_step_index": step_index + 1,
        }

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    llm = llm or get_chat_llm()
    retriever = retriever or get_retriever()
    tools = tools if tools is not None else load_mcp_tools()

    planner = make_planner(llm)
    supervisor = make_supervisor(llm)
    rag_agent = make_rag_agent(retriever, llm)
    mcp_tools = make_mcp_node(tools, llm)
    synthesizer = make_synthesizer(llm)

    builder = StateGraph(AnalystState)

    builder.add_node("planner", planner)
    builder.add_node("supervisor", supervisor)
    builder.add_node("rag_agent", rag_agent)
    builder.add_node("mcp_tools", mcp_tools)
    builder.add_node("synthesizer", synthesizer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            RAG: "rag_agent",
            MCP: "mcp_tools",
            SYNTH: "synthesizer",
        },
    )
    builder.add_edge("rag_agent", "supervisor")
    builder.add_edge("mcp_tools", "supervisor")
    builder.add_edge("synthesizer", END)

    return builder.compile()