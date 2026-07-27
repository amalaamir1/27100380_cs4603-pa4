"""ChatModel wrapper for the UC Function Document Analyst."""

from __future__ import annotations

import os

import mlflow
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import ChatCompletionResponse

from agent.graph_uc import build_graph, load_uc_tools


MODEL_ENDPOINT = os.environ.get(
    "DATABRICKS_MODEL",
    "databricks-meta-llama-3-3-70b-instruct",
)

VECTOR_SEARCH_INDEX = os.environ.get(
    "VECTOR_SEARCH_INDEX",
    "cs4603.default.s27100380_analyst_index",
)


llm = ChatDatabricks(
    endpoint=MODEL_ENDPOINT,
    temperature=0.0,
)

vector_store = DatabricksVectorSearch(
    index_name=VECTOR_SEARCH_INDEX,
    columns=["chunk_id", "source", "page"],
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4},
)

uc_tools = load_uc_tools()

graph = build_graph(
    llm=llm,
    retriever=retriever,
    tools=uc_tools,
)


class DocumentAnalystChatModel(ChatModel):
    """Expose the LangGraph agent through an OpenAI-compatible interface."""

    def predict(
        self,
        context,
        messages,
        params=None,
    ) -> ChatCompletionResponse:
        graph_messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

        result = graph.invoke(
            {
                "messages": graph_messages,
            }
        )

        final_message = result["messages"][-1]

        if hasattr(final_message, "content"):
            answer = final_message.content
        else:
            answer = final_message["content"]

        return ChatCompletionResponse(
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                    },
                    "finish_reason": "stop",
                }
            ],
            model=MODEL_ENDPOINT,
        )


mlflow.models.set_model(DocumentAnalystChatModel())