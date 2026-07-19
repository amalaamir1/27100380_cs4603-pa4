from databricks.sdk import WorkspaceClient
from databricks_langchain import DatabricksVectorSearch

from config import get_settings

TEXT_COLUMN = "chunk_to_retrieve"
CITATION_COLUMNS = ["chunk_id", "source", "page"]


def get_vector_store():
    settings = get_settings()

    workspace_client = WorkspaceClient(
        host=settings["host"],
        token=settings["token"],
    )

    return DatabricksVectorSearch(
        index_name=settings["vs_index"],
        workspace_client=workspace_client,
        columns=CITATION_COLUMNS,
    )

def get_retriever(k: int = 4):
    return get_vector_store().as_retriever(
        search_kwargs={"k": k}
    )