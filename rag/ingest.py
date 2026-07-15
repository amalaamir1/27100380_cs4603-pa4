"""Corpus ingestion into Databricks Vector Search (Task 0.3).

Run this module from a Databricks notebook.  Document parsing and semantic
chunking use Databricks SQL AI functions and therefore require Spark compute
that supports ``ai_parse_document`` and ``ai_prep_search``.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any


def _sql_identifier(value: str, *, label: str) -> str:
    """Validate a two- or three-part Unity Catalog identifier."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){1,2}", value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise OSError(f"Missing required environment variable: {name}")
    return value


def _parsed_table_name(chunks_table: str) -> str:
    parts = chunks_table.split(".")
    parts[-1] = f"{parts[-1]}_parsed_documents"
    return ".".join(parts)


def build_chunks_table(spark, volume_path: str, chunks_table: str) -> None:
    """Parse the PDF and replace the Delta source table with semantic chunks.

    Re-running this function is safe: both intermediate and final tables use
    ``INSERT OVERWRITE``, so the same PDF does not create duplicate chunks.
    """
    chunks_table = _sql_identifier(chunks_table, label="chunks table")
    parsed_table = _sql_identifier(
        _parsed_table_name(chunks_table), label="parsed documents table"
    )
    escaped_path = volume_path.replace("'", "''")

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {parsed_table} (
            path STRING,
            parsed VARIANT
        )
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """
    )

    spark.sql(
        f"""
        INSERT OVERWRITE {parsed_table}
        SELECT
            path,
            ai_parse_document(content) AS parsed
        FROM READ_FILES('{escaped_path}', format => 'binaryFile')
        """
    )

    parsed_count = spark.table(parsed_table).count()
    if parsed_count == 0:
        raise RuntimeError(f"No documents were found at {volume_path!r}")
    print(f"Parsed {parsed_count} document(s) into {parsed_table}.")

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {chunks_table} (
            chunk_id STRING,
            chunk_to_retrieve STRING,
            chunk_to_embed STRING,
            source STRING,
            page INT
        )
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """
    )

    spark.sql(
        f"""
        INSERT OVERWRITE {chunks_table}
        SELECT
            chunk.value:chunk_id::STRING AS chunk_id,
            chunk.value:chunk_to_retrieve::STRING AS chunk_to_retrieve,
            chunk.value:chunk_to_embed::STRING AS chunk_to_embed,
            regexp_extract(path, '[^/]+$', 0) AS source,
            chunk.value:pages[0]:page_id::INT + 1 AS page
        FROM (
            SELECT path, ai_prep_search(parsed) AS result
            FROM {parsed_table}
        ) AS prepped,
        LATERAL variant_explode(result:document.contents) AS chunk
        """
    )

    chunk_count = spark.table(chunks_table).count()
    if chunk_count == 0:
        raise RuntimeError("ai_prep_search produced no chunks")

    duplicate_count = spark.sql(
        f"""
        SELECT COUNT(*) AS duplicate_count
        FROM (
            SELECT chunk_id
            FROM {chunks_table}
            GROUP BY chunk_id
            HAVING COUNT(*) > 1
        )
        """
    ).first()["duplicate_count"]
    if duplicate_count:
        raise RuntimeError(f"Found {duplicate_count} duplicate chunk_id value(s)")

    print(f"Created {chunk_count} unique chunks in {chunks_table}.")


def _index_state(description: dict[str, Any]) -> tuple[bool, str]:
    """Read readiness across the response shapes used by AI Search clients."""
    status = description.get("status", {})
    ready = status.get("ready")
    state = str(
        status.get("detailed_state")
        or status.get("status")
        or description.get("detailed_state")
        or "UNKNOWN"
    )
    return ready is True or state.upper() in {"READY", "ONLINE"}, state


def create_index(*, wait_timeout: int = 1800, poll_interval: int = 15):
    """Create the STANDARD endpoint and TRIGGERED Delta Sync index, then wait.

    Names come from the PA4 environment variables so this is the same index
    later consumed by ``rag.store`` and by the deployed graph.
    """
    from databricks.ai_search.client import AISearchClient

    endpoint_name = _required_env("VECTOR_SEARCH_ENDPOINT")
    index_name = _sql_identifier(_required_env("VECTOR_SEARCH_INDEX"), label="index")
    source_table = _sql_identifier(_required_env("SOURCE_TABLE"), label="source table")
    embeddings_endpoint = os.environ.get(
        "EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"
    ).strip()

    client = AISearchClient()
    endpoint_names = {
        endpoint["name"]
        for endpoint in client.list_endpoints().get("endpoints", [])
    }
    if endpoint_name not in endpoint_names:
        client.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
        print(f"Created STANDARD endpoint {endpoint_name!r}.")
    else:
        print(f"Endpoint {endpoint_name!r} already exists.")

    existing_indexes = {
        item["name"]
        for item in client.list_indexes(name=endpoint_name).get("vector_indexes", [])
    }
    if index_name not in existing_indexes:
        client.create_delta_sync_index(
            endpoint_name=endpoint_name,
            source_table_name=source_table,
            index_name=index_name,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_source_column="chunk_to_retrieve",
            embedding_model_endpoint_name=embeddings_endpoint,
        )
        print(f"Created TRIGGERED Delta Sync index {index_name!r}.")
    else:
        print(f"Index {index_name!r} already exists; triggering a sync.")

    index = client.get_index(endpoint_name=endpoint_name, index_name=index_name)
    if index_name in existing_indexes:
        index.sync()

    deadline = time.monotonic() + wait_timeout
    while True:
        description = index.describe()
        ready, state = _index_state(description)
        print(f"Index state: {state}")
        if ready:
            print(f"Index {index_name!r} is READY.")
            return index
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Index {index_name!r} did not become READY within {wait_timeout} seconds"
            )
        time.sleep(poll_interval)


def verify_index(index, *, query: str = "What was the net income in 2023?") -> dict:
    """Run the Task 0.3 similarity-search acceptance check."""
    response = index.similarity_search(
        query_text=query,
        columns=["chunk_to_retrieve", "source", "page", "chunk_id"],
        num_results=3,
    )
    rows = response.get("result", {}).get("data_array", [])
    if not rows:
        raise RuntimeError("The index is READY but returned no similarity-search results")
    print(f"Similarity search returned {len(rows)} result(s).")
    for row in rows:
        print(row)
    return response


def ingest(spark, volume_path: str):
    """Run the complete Part 0.3 pipeline and return the ready index handle."""
    source_table = _required_env("SOURCE_TABLE")
    build_chunks_table(spark, volume_path, source_table)
    index = create_index()
    verify_index(index)
    return index
