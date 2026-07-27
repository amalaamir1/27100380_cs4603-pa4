"""Document Analyst graph backed by governed Unity Catalog functions."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from databricks_langchain import UCFunctionToolkit
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import json
from typing import Any

from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import (
    MCP,
    RAG,
    SYNTH,
    make_supervisor,
    route_from_supervisor,
)
from agent.synthesizer import make_synthesizer
from config import get_chat_llm
from rag.store import get_retriever


UC_FUNCTION_NAMES = [
    "cs4603.default.growth_rate",
    "cs4603.default.percentage_change",
    "cs4603.default.compare_values",
    "cs4603.default.to_billions",
]

UC_STEP_PROMPT = """Execute one numerical plan step using the supplied governed
Unity Catalog functions. Use the current step and relevant results from earlier
steps. Select and call exactly one function. Pass explicit numeric arguments
with consistent units and decimal rates. Do not perform the calculation
mentally. Return the function result.
"""


def _run_async_synchronously(coroutine):
    """Run an asynchronous operation from synchronous Python."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


def _message_text(message) -> str:
    """Normalize a tool response into text."""
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


UC_PLANNER_PROMPT = """You are the planner for a financial document analyst.

Break the user's query into 2 to 5 sequential atomic steps. Put document
retrieval before calculations that depend on retrieved values.

Available calculation functions:
- growth_rate(start_value, rate, years) directly returns the final projected
  value after compound growth.
- percentage_change(old_value, new_value) returns the percentage change.
- compare_values(first_value, second_value) compares two values.
- to_billions(amount) converts base units into billions.

Important:
- Each calculation step must require exactly one function call.
- Never create a separate step to calculate a compound-growth factor and
  another step to apply that factor.
- For compound growth, retrieve the starting value and then use exactly one
  growth_rate step to calculate the final projected value.
- Preserve units between retrieval and calculation.

Return only a valid JSON list of step strings.
"""


def _planner_message_text(message: Any) -> str:
    """Extract text from a LangChain message or message dictionary."""
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    return content if isinstance(content, str) else str(content)


def make_uc_planner(llm):
    """Create a planner aware of the UC functions' exact semantics."""

    def planner(state: AnalystState) -> dict:
        messages = state.get("messages", [])

        if not messages:
            raise ValueError("Planner requires at least one message")

        user_query = _planner_message_text(messages[-1]).strip()

        if not user_query:
            raise ValueError("Planner received an empty query")

        response = llm.invoke(
            [
                SystemMessage(content=UC_PLANNER_PROMPT),
                HumanMessage(content=user_query),
            ]
        )

        response_text = _planner_message_text(response).strip()

        if response_text.startswith("```"):
            lines = response_text.splitlines()

            if lines and lines[0].strip().lower() in {"```", "```json"}:
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            response_text = "\n".join(lines).strip()

        try:
            plan = json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            plan = None

        valid_plan = (
            isinstance(plan, list)
            and 2 <= len(plan) <= 5
            and all(
                isinstance(step, str) and step.strip()
                for step in plan
            )
        )

        if not valid_plan:
            plan = [user_query]
        else:
            plan = [step.strip() for step in plan]

        return {
            "plan": plan,
            "current_step_index": 0,
            "step_results": [],
        }

    return planner

def load_uc_tools(function_names=None):
    """Load governed UC functions as LangChain tools."""
    selected_functions = function_names or UC_FUNCTION_NAMES

    toolkit = UCFunctionToolkit(
        function_names=list(selected_functions),
    )

    return toolkit.tools


def _invoke_tool(tool, arguments):
    """Invoke either a synchronous or asynchronous LangChain tool."""
    try:
        return tool.invoke(arguments)
    except NotImplementedError:
        if not hasattr(tool, "ainvoke"):
            raise

        return _run_async_synchronously(tool.ainvoke(arguments))


def make_uc_tools_node(tools, llm):
    """Create a graph node that executes one Unity Catalog function."""
    if not tools:
        raise ValueError("At least one Unity Catalog function is required")

    tools_by_name = {tool.name: tool for tool in tools}
    tool_llm = llm.bind_tools(tools)

    def uc_tools(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        step_index = state.get("current_step_index", 0)

        if step_index >= len(plan):
            raise ValueError("UC tools node requires an unfinished plan step")

        current_step = plan[step_index]
        previous_results = state.get("step_results", [])

        previous_context = "\n".join(
            f"Step {index + 1}: {result}"
            for index, result in enumerate(previous_results)
        ) or "None"

        decision = tool_llm.invoke(
            [
                SystemMessage(content=UC_STEP_PROMPT),
                HumanMessage(
                    content=(
                        f"Current numerical step:\n{current_step}\n\n"
                        f"Results from earlier steps:\n{previous_context}"
                    )
                ),
            ]
        )

        tool_calls = getattr(decision, "tool_calls", [])

        if len(tool_calls) != 1:
            raise ValueError(
                "UC calculation step must produce exactly one function call; "
                f"received {len(tool_calls)}"
            )

        call = tool_calls[0]
        tool_name = call.get("name")
        tool_arguments = call.get("args", {})

        if tool_name not in tools_by_name:
            raise ValueError(
                f"LLM selected an unknown UC function: {tool_name!r}"
            )

        result = _invoke_tool(
            tools_by_name[tool_name],
            tool_arguments,
        )

        result_text = _message_text(result)

        if not result_text:
            result_text = "Unity Catalog function returned no result"

        return {
            "step_results": [*previous_results, result_text],
            "current_step_index": step_index + 1,
        }

    return uc_tools


def build_graph(
    llm=None,
    retriever=None,
    tools=None,
):
    """Build the PA4 graph using UC Functions instead of MCP tools."""
    llm = llm or get_chat_llm()
    retriever = retriever or get_retriever()
    tools = tools if tools is not None else load_uc_tools()

    planner = make_uc_planner(llm)
    supervisor = make_supervisor(llm)
    rag_agent = make_rag_agent(retriever, llm)
    uc_tools = make_uc_tools_node(tools, llm)
    synthesizer = make_synthesizer(llm)

    builder = StateGraph(AnalystState)

    builder.add_node("planner", planner)
    builder.add_node("supervisor", supervisor)
    builder.add_node("rag_agent", rag_agent)
    builder.add_node("uc_tools", uc_tools)
    builder.add_node("synthesizer", synthesizer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")

    # The existing supervisor still returns the internal MCP route constant
    # for calculations. Only its destination changes to the UC-backed node.
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            RAG: "rag_agent",
            MCP: "uc_tools",
            SYNTH: "synthesizer",
        },
    )

    builder.add_edge("rag_agent", "supervisor")
    builder.add_edge("uc_tools", "supervisor")
    builder.add_edge("synthesizer", END)

    return builder.compile()