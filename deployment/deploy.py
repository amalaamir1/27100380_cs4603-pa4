"""Log and register the PA4 Document Analyst model (Task 2.2)."""
from __future__ import annotations
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
import os
from datetime import timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    
)

PIP_REQUIREMENTS = [
    "mlflow>=2.16.0",
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-openai>=0.2.0",
    "databricks-langchain>=0.1.0",
    "databricks-ai-search",
    "databricks-vectorsearch>=0.40",
    "databricks-sdk>=0.23.0",
    "langchain-mcp-adapters>=0.0.5",
    "mcp>=1.0.0",
    "python-dotenv>=1.0.0",
    "httpx>=0.27.0",
    "openai>=1.40.0",
    "typing_extensions>=4.15.0",
]


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def log_and_register() -> tuple[str, str]:
    """Log the models-from-code graph and register it in Unity Catalog."""
    root = Path(__file__).resolve().parents[1]
    model_file = Path(__file__).resolve().parent / "agent_model.py"

    if not model_file.is_file():
        raise FileNotFoundError(f"Model definition not found: {model_file}")

    catalog = _require("UC_CATALOG")
    schema = _require("UC_SCHEMA")
    model_name = os.environ.get("UC_MODEL_NAME", "s27100380_document_analyst")
    uc_name = f"{catalog}.{schema}.{model_name}"

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if not experiment:
        user_name = WorkspaceClient().current_user.me().user_name
        experiment = f"/Users/{user_name}/pa4-document-analyst"
    mlflow.set_experiment(experiment)

    input_example = {
    "messages": [
        {
            "role": "user",
            "content": "What was the revenue in 2023?",
        }
    ]
    }

    output_example = {
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "The 2023 revenue was 100 million dollars "
                    "[source: annual_report.pdf, p.7]."
                ),
            }
        ],
        "plan": ["Find the 2023 revenue in the document"],
        "current_step_index": 1,
        "step_results": [
            "The 2023 revenue was 100 million dollars "
            "[source: annual_report.pdf, p.7]."
        ],
        "next_agent": "synthesizer",
        "final_answer": (
            "The 2023 revenue was 100 million dollars "
            "[source: annual_report.pdf, p.7]."
        ),
    }

    signature = mlflow.models.infer_signature(
        input_example,
        output_example,
    )


    with mlflow.start_run(run_name="pa4-document-analyst"):
        model_info = mlflow.langchain.log_model(
            lc_model=str(model_file),
            name="agent",
            code_paths=[
                str(root / "agent"),
                str(root / "rag"),
                str(root / "tools"),
                str(root / "config.py"),
            ],
            pip_requirements=PIP_REQUIREMENTS,
            input_example=input_example,
            signature=signature,
        )

    registered = mlflow.register_model(model_info.model_uri, uc_name)
    version = str(registered.version)

    print(f"Model URI: {model_info.model_uri}")
    print(f"Registered model: {uc_name}")
    print(f"Registered version: {version}")
    return uc_name, version
   
def create_or_update_endpoint(uc_name: str, version: str) -> str:
    """Create or update the endpoint and wait until its configuration is ready."""
    endpoint_name = os.environ.get(
        "SERVING_ENDPOINT_NAME", "s27100380-document-analyst"
    )
    secret_scope = os.environ.get("SECRET_SCOPE", "cs4603-deploy")

    # These non-secret settings identify the managed Vector Search resources.
    vector_search_endpoint = _require("VECTOR_SEARCH_ENDPOINT")
    vector_search_index = _require("VECTOR_SEARCH_INDEX")
    embeddings_endpoint = os.environ.get(
        "EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"
    )

    served_entity = ServedEntityInput(
        name=f"s27100380-document-analyst-{version}",
        entity_name=uc_name,
        entity_version=version,
        workload_size="Small",
        scale_to_zero_enabled=True,
        environment_vars={
            "DATABRICKS_HOST": (
                f"{{{{secrets/{secret_scope}/DATABRICKS_HOST}}}}"
            ),
            "DATABRICKS_TOKEN": (
                f"{{{{secrets/{secret_scope}/DATABRICKS_TOKEN}}}}"
            ),
            "DATABRICKS_MODEL": (
                f"{{{{secrets/{secret_scope}/DATABRICKS_MODEL}}}}"
            ),
            "VECTOR_SEARCH_ENDPOINT": vector_search_endpoint,
            "VECTOR_SEARCH_INDEX": vector_search_index,
            "EMBEDDINGS_ENDPOINT": embeddings_endpoint,
        },
    )

    workspace = WorkspaceClient()
    timeout = timedelta(minutes=30)

    try:
        workspace.serving_endpoints.get(endpoint_name)
    except NotFound:
        print(f"Creating serving endpoint '{endpoint_name}'...")
        endpoint = workspace.serving_endpoints.create_and_wait(
            name=endpoint_name,
            config=EndpointCoreConfigInput(served_entities=[served_entity]),
            timeout=timeout,
        )
    else:
        print(f"Updating serving endpoint '{endpoint_name}' to version {version}...")
        endpoint = workspace.serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_entities=[served_entity],
            timeout=timeout,
        )

    ready = getattr(getattr(endpoint, "state", None), "ready", None)
    print(f"Endpoint state: {ready}")
    if str(ready).split(".")[-1] != "READY":
        raise RuntimeError(
            f"Endpoint '{endpoint_name}' did not reach READY; state={ready}"
        )

    host = workspace.config.host.rstrip("/")
    endpoint_page = f"{host}/ml/endpoints/{endpoint_name}"
    invocation_url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
    print(f"Endpoint page: {endpoint_page}")
    print(f"Invocation URL: {invocation_url}")
    return invocation_url



if __name__ == "__main__":
    log_and_register()
