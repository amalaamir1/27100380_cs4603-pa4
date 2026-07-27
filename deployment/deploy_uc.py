"""Deploy the UC Function version of the Document Analyst."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)


PIP_REQUIREMENTS = [
    "mlflow>=2.16.0",
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-openai>=0.2.0",
    "databricks-langchain>=0.1.0",
    "databricks-vectorsearch>=0.40",
    "databricks-sdk>=0.23.0",
    "unitycatalog-ai[databricks]",
    "unitycatalog-langchain",
    "python-dotenv>=1.0.0",
    "typing_extensions>=4.15.0",
]


UC_FUNCTIONS = [
    "cs4603.default.growth_rate",
    "cs4603.default.percentage_change",
    "cs4603.default.compare_values",
    "cs4603.default.to_billions",
]


def require_env(name: str) -> str:
    value = os.environ.get(name)

    if not value:
        raise OSError(f"Missing required environment variable: {name}")

    return value


def log_and_register_uc() -> tuple[str, str]:
    """Log and register the UC-backed graph."""

    root = Path(__file__).resolve().parents[1]
    model_file = root / "deployment" / "agent_model_uc.py"

    catalog = os.environ.get("UC_CATALOG", "cs4603")
    schema = os.environ.get("UC_SCHEMA", "default")
    model_name = os.environ.get(
        "UC_TOOLS_MODEL_NAME",
        "s27100380_document_analyst_uc",
    )
    registered_model_name = f"{catalog}.{schema}.{model_name}"

    model_endpoint = os.environ.get(
        "DATABRICKS_MODEL",
        "databricks-meta-llama-3-3-70b-instruct",
    )
    vector_index = require_env("VECTOR_SEARCH_INDEX")

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    user_name = WorkspaceClient().current_user.me().user_name
    mlflow.set_experiment(
        f"/Users/{user_name}/pa4-extra-credit-uc-tools"
    )

    input_example = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "What was Meridian's FY2023 net revenue, and what "
                    "would it be after 3 years of 8% compound growth?"
                ),
            }
        ]
    }

    output_example = {
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Meridian's projected revenue is approximately "
                    "¥21,301.73 billion."
                ),
            }
        ],
        "plan": [
            "Retrieve Meridian's FY2023 net revenue",
            "Calculate its value after 3 years of 8% compound growth",
        ],
        "current_step_index": 2,
        "step_results": [
            "Meridian's FY2023 net revenue was ¥16,910 billion.",
            '{"format": "SCALAR", "value": "21.301729920000003"}',
        ],
        "next_agent": "synthesizer",
        "final_answer": (
            "Meridian's projected revenue is approximately "
            "¥21,301.73 billion."
        ),
    }

    signature = mlflow.models.infer_signature(
        input_example,
        output_example,
    )

    resources = [
        DatabricksServingEndpoint(endpoint_name=model_endpoint),
        DatabricksVectorSearchIndex(index_name=vector_index),
        *[
            DatabricksFunction(function_name=function_name)
            for function_name in UC_FUNCTIONS
        ],
    ]

    with mlflow.start_run(run_name="pa4-uc-function-agent"):
        model_info = mlflow.pyfunc.log_model(
            python_model=str(model_file),
            name="agent",
            code_paths=[
                str(root / "agent"),
                str(root / "rag"),
                str(root / "config.py"),
            ],
            pip_requirements=PIP_REQUIREMENTS,
            resources=resources,
            input_example=input_example,
        )

    registered = mlflow.register_model(
        model_info.model_uri,
        registered_model_name,
    )

    version = str(registered.version)

    print("Model URI:", model_info.model_uri)
    print("Registered model:", registered_model_name)
    print("Registered version:", version)

    return registered_model_name, version


def deploy_uc_agent():
    """Deploy using automatic Databricks resource authentication."""

    model_name, version = log_and_register_uc()

    deployment = agents.deploy(
        model_name=model_name,
        model_version=int(version),
        endpoint_name=os.environ.get(
            "UC_TOOLS_ENDPOINT_NAME",
            "s27100380-document-analyst-uc",
        ),
        scale_to_zero=True,
        workload_size="Small",
        environment_vars={
            "DATABRICKS_MODEL": os.environ.get(
                "DATABRICKS_MODEL",
                "databricks-meta-llama-3-3-70b-instruct",
            ),
            "VECTOR_SEARCH_INDEX": require_env(
                "VECTOR_SEARCH_INDEX"
            ),
        },
    )

    print("Endpoint:", deployment.endpoint_name)
    print("Query URL:", deployment.query_endpoint)
    print("Review App:", deployment.review_app_url)

    return deployment


if __name__ == "__main__":
    deploy_uc_agent()