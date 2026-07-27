"""Build governed structured financial tables for Part 2."""

from __future__ import annotations

import os

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)


CATALOG = os.environ.get("UC_CATALOG", "cs4603")
SCHEMA = os.environ.get("UC_SCHEMA", "default")

SEGMENT_TABLE = (
    f"{CATALOG}.{SCHEMA}."
    "s27100380_meridian_segment_financials"
)

INCOME_TABLE = (
    f"{CATALOG}.{SCHEMA}."
    "s27100380_meridian_income_statement"
)


def build_tables(spark: SparkSession) -> dict[str, str]:
    """Create and document the structured Meridian Delta tables."""

    segment_schema = StructType(
        [
            StructField("fiscal_year", IntegerType(), False),
            StructField("segment", StringType(), False),
            StructField("revenue_yen", LongType(), False),
            StructField(
                "operating_income_yen",
                LongType(),
                False,
            ),
            StructField("units_sold", LongType(), True),
            StructField("source_note", StringType(), False),
        ]
    )

    segment_rows = [
        # FY2023 values reconcile to ¥16.91T revenue and
        # ¥1.124T operating income.
        (
            2023,
            "Automobile",
            12_900_000_000_000,
            560_000_000_000,
            4_070_000,
            "Derived from the FY2023 Meridian annual report.",
        ),
        (
            2023,
            "Motorcycle",
            2_510_000_000_000,
            360_000_000_000,
            18_500_000,
            "Derived from the FY2023 Meridian annual report.",
        ),
        (
            2023,
            "Financial Services",
            1_100_000_000_000,
            180_000_000_000,
            None,
            "Derived from the FY2023 Meridian annual report.",
        ),
        (
            2023,
            "Power Products & Other",
            400_000_000_000,
            24_000_000_000,
            None,
            "Derived from the FY2023 Meridian annual report.",
        ),
        # FY2022 values are synthetic but internally consistent.
        (
            2022,
            "Automobile",
            11_850_000_000_000,
            480_000_000_000,
            3_800_000,
            "Synthetic prior-year value for YoY analysis.",
        ),
        (
            2022,
            "Motorcycle",
            2_250_000_000_000,
            310_000_000_000,
            17_000_000,
            "Synthetic prior-year value for YoY analysis.",
        ),
        (
            2022,
            "Financial Services",
            1_000_000_000_000,
            160_000_000_000,
            None,
            "Synthetic prior-year value for YoY analysis.",
        ),
        (
            2022,
            "Power Products & Other",
            360_000_000_000,
            20_000_000_000,
            None,
            "Synthetic prior-year value for YoY analysis.",
        ),
    ]

    segment_df = spark.createDataFrame(
        segment_rows,
        schema=segment_schema,
    )

    (
        segment_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(SEGMENT_TABLE)
    )

    income_schema = StructType(
        [
            StructField("fiscal_year", IntegerType(), False),
            StructField("line_item", StringType(), False),
            StructField("amount_yen", LongType(), False),
            StructField("source_note", StringType(), False),
        ]
    )

    income_rows = [
        (
            2023,
            "Net Revenue",
            16_910_000_000_000,
            "FY2023 Meridian annual report.",
        ),
        (
            2023,
            "Operating Profit",
            1_124_000_000_000,
            "FY2023 segment operating-income total.",
        ),
        (
            2023,
            "Net Income",
            1_107_000_000_000,
            "FY2023 Meridian annual report.",
        ),
        (
            2022,
            "Net Revenue",
            15_460_000_000_000,
            "Synthetic prior-year value for YoY analysis.",
        ),
        (
            2022,
            "Operating Profit",
            970_000_000_000,
            "Synthetic prior-year value for YoY analysis.",
        ),
        (
            2022,
            "Net Income",
            950_000_000_000,
            "Synthetic prior-year value for YoY analysis.",
        ),
    ]

    income_df = spark.createDataFrame(
        income_rows,
        schema=income_schema,
    )

    (
        income_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(INCOME_TABLE)
    )

    # Table comments
    spark.sql(
        f"""
        COMMENT ON TABLE {SEGMENT_TABLE} IS
        'Meridian segment financial results by fiscal year.
        Monetary values are stored as raw Japanese yen.'
        """
    )

    spark.sql(
        f"""
        COMMENT ON TABLE {INCOME_TABLE} IS
        'Meridian income-statement values by fiscal year.
        Monetary values are stored as raw Japanese yen.'
        """
    )

    # Segment table column comments
    segment_comments = {
        "fiscal_year": (
            "Four-digit fiscal year. FY means fiscal_year."
        ),
        "segment": "Meridian operating business segment.",
        "revenue_yen": (
            "Segment revenue in raw Japanese yen."
        ),
        "operating_income_yen": (
            "Segment operating income in raw Japanese yen."
        ),
        "units_sold": (
            "Units sold where applicable; otherwise NULL."
        ),
        "source_note": (
            "Whether the value came from the report or was synthesized."
        ),
    }

    for column, comment in segment_comments.items():
        spark.sql(
            f"""
            ALTER TABLE {SEGMENT_TABLE}
            ALTER COLUMN {column}
            COMMENT '{comment}'
            """
        )

    # Income-statement column comments
    income_comments = {
        "fiscal_year": (
            "Four-digit fiscal year. FY means fiscal_year."
        ),
        "line_item": (
            "Income-statement metric such as Net Revenue or Net Income."
        ),
        "amount_yen": (
            "Financial amount in raw Japanese yen."
        ),
        "source_note": (
            "Whether the value came from the report or was synthesized."
        ),
    }

    for column, comment in income_comments.items():
        spark.sql(
            f"""
            ALTER TABLE {INCOME_TABLE}
            ALTER COLUMN {column}
            COMMENT '{comment}'
            """
        )

    print(f"Created: {SEGMENT_TABLE}")
    print(f"Created: {INCOME_TABLE}")

    return {
        "segment_table": SEGMENT_TABLE,
        "income_table": INCOME_TABLE,
    }