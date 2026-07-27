"""Register PA4 calculation tools as governed Unity Catalog functions."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

load_dotenv()

CATALOG = os.getenv("UC_CATALOG", "cs4603")
SCHEMA = os.getenv("UC_SCHEMA", "default")


def growth_rate(
    start_value: float,
    rate: float,
    years: int,
) -> float:
    """Project a value using compound annual growth.

    Args:
        start_value: Initial value before applying growth.
        rate: Annual growth rate as a decimal, such as 0.08 for 8 percent.
        years: Number of complete years over which growth is compounded.

    Returns:
        The projected value after compound growth.
    """
    return start_value * (1.0 + rate) ** years


def percentage_change(
    old_value: float,
    new_value: float,
) -> float:
    """Calculate percentage change from an old value to a new value.

    Args:
        old_value: Original baseline value, which must not be zero.
        new_value: Updated value being compared with the baseline.

    Returns:
        Percentage change. Positive means an increase and negative means
        a decrease.

    Raises:
        ValueError: If old_value is zero.
    """
    if old_value == 0:
        raise ValueError("old_value must not be zero")

    return ((new_value - old_value) / abs(old_value)) * 100.0


def compare_values(
    first_value: float,
    second_value: float,
) -> str:
    """Compare two values and describe their absolute difference.

    Args:
        first_value: First numeric value to compare.
        second_value: Second numeric value to compare.

    Returns:
        A description identifying the larger value and their difference.
    """
    if first_value == second_value:
        return f"{first_value:g} and {second_value:g} are equal"

    larger = max(first_value, second_value)
    smaller = min(first_value, second_value)
    difference = larger - smaller

    return (
        f"{larger:g} is larger than {smaller:g} "
        f"by an absolute difference of {difference:g}"
    )


PYTHON_FUNCTIONS = [
    growth_rate,
    percentage_change,
    compare_values,
]


def register_python_functions(
    client: DatabricksFunctionClient | None = None,
) -> list[str]:
    """Create or replace all three Python functions in Unity Catalog."""
    function_client = client or DatabricksFunctionClient()
    registered_names = []

    for function in PYTHON_FUNCTIONS:
        function_client.create_python_function(
            func=function,
            catalog=CATALOG,
            schema=SCHEMA,
            replace=True,
        )

        full_name = f"{CATALOG}.{SCHEMA}.{function.__name__}"
        registered_names.append(full_name)
        print(f"Registered: {full_name}")

    return registered_names


def verify_python_functions(
    client: DatabricksFunctionClient | None = None,
) -> dict[str, Any]:
    """Execute every registered Python function directly."""
    function_client = client or DatabricksFunctionClient()

    test_cases = {
        f"{CATALOG}.{SCHEMA}.growth_rate": {
            "start_value": 16.91,
            "rate": 0.08,
            "years": 3,
        },
        f"{CATALOG}.{SCHEMA}.percentage_change": {
            "old_value": 100.0,
            "new_value": 125.0,
        },
        f"{CATALOG}.{SCHEMA}.compare_values": {
            "first_value": 16.91,
            "second_value": 18.60,
        },
    }

    results = {}

    for function_name, parameters in test_cases.items():
        result = function_client.execute_function(
            function_name,
            parameters,
        )
        results[function_name] = result

        print(f"\nFunction: {function_name}")
        print(f"Parameters: {parameters}")
        print(f"Result: {result}")

    return results


if __name__ == "__main__":
    register_python_functions()
    verify_python_functions()