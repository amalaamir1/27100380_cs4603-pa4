from __future__ import annotations

import os
from pathlib import Path

import mlflow

from agent.graph import build_graph, load_mcp_tools
from config import get_chat_llm
from rag.store import get_retriever
import langchain_mcp_adapters.sessions as mcp_sessions
from mcp.client.stdio import stdio_client as sdk_stdio_client

# Validate every value required while the serving container imports this file.
_REQUIRED_ENV_VARS = (
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_MODEL",
    "VECTOR_SEARCH_ENDPOINT",
    "VECTOR_SEARCH_INDEX",
)

_missing = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]
if _missing:
    raise EnvironmentError(
        "Missing required environment variables: "
        f"{', '.join(_missing)}. Configure them locally or inject them into "
        "the Databricks serving endpoint before the model is loaded."
    )


# The MCP server is bundled through code_paths=["tools"] in Task 2.2.
_model_dir = Path(__file__).resolve().parent

_mcp_candidates = (
    _model_dir / "code" / "tools" / "mcp_server.py",  # serving
    _model_dir.parent / "tools" / "mcp_server.py",    # local repo
    Path.cwd() / "tools" / "mcp_server.py",
)

_mcp_server = next(
    (path for path in _mcp_candidates if path.is_file()),
    None,
)

if _mcp_server is None:
    raise FileNotFoundError(
        "Bundled MCP server not found. Checked: "
        + ", ".join(str(path) for path in _mcp_candidates)
    )
# Databricks replaces stderr with StreamToLogger, which has no fileno().
# MCP subprocess creation requires a real OS-backed stream.
_mcp_errlog = open(os.devnull, "w", encoding="utf-8")


def _serving_stdio_client(server):
    return sdk_stdio_client(server, errlog=_mcp_errlog)


# Patch the reference used by langchain-mcp-adapters.
mcp_sessions.stdio_client = _serving_stdio_client

# Build production dependencies once when MLflow loads the model artifact.
llm = get_chat_llm(temperature=0.0)
retriever = get_retriever(k=4)
tools = load_mcp_tools(str(_mcp_server))

graph = build_graph(
    llm=llm,
    retriever=retriever,
    tools=tools,
)

# Models-from-code discovers the object to serve through this call.
mlflow.models.set_model(graph)