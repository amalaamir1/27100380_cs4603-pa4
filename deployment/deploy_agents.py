"""Bonus B: deploy the PA4 Document Analyst with databricks-agents."""

from __future__ import annotations

import os

from databricks import agents

from deployment.deploy import log_and_register


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise OSError(f"Missing required environment variable: {name}")
    return value


def deploy_with_agents():
    """Log/register the model, then deploy it through the Agent Framework."""
    uc_name, version = log_and_register()

    # DATABRICKS_HOST and DATABRICKS_TOKEN are intentionally not supplied here.
    # agents.deploy() manages the endpoint's Databricks authentication.
    runtime_config = {
        "DATABRICKS_MODEL": _require("DATABRICKS_MODEL"),
        "VECTOR_SEARCH_ENDPOINT": _require("VECTOR_SEARCH_ENDPOINT"),
        "VECTOR_SEARCH_INDEX": _require("VECTOR_SEARCH_INDEX"),
        "EMBEDDINGS_ENDPOINT": os.environ.get(
            "EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"
        ),
    }

    deployment = agents.deploy(
        model_name=uc_name,
        model_version=int(version),
        endpoint_name=os.environ.get(
            "AGENTS_ENDPOINT_NAME", "s27100380-document-analyst-agents"
        ),
        scale_to_zero=True,
        workload_size="Small",
        environment_vars=runtime_config,
    )

    print(f"Registered model: {uc_name}")
    print(f"Model version: {version}")
    print(f"Endpoint name: {deployment.endpoint_name}")
    print(f"Query endpoint: {deployment.query_endpoint}")
    print(f"Review App URL: {deployment.review_app_url}")
    return deployment


if __name__ == "__main__":
    deploy_with_agents()
