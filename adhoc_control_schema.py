"""
adhoc_control_schema.py
───────────────────────
Example: how to create or populate the ad-hoc control dataset
that feeds RIDs into Mode B (dynamic runtime profiling).

This is a separate transform that writes the control table.
The profiler's ad-hoc mode reads this as its single declared input,
then uses ctx._foundry.input(rid) to dynamically read each listed dataset.
"""

from transforms.api import transform_df, Input, Output
import pyspark.sql.functions as F
import pyspark.sql.types as T


# Control table schema
CONTROL_SCHEMA = T.StructType([
    T.StructField("rid", T.StringType(), nullable=False),
    T.StructField("path", T.StringType(), nullable=False),
    T.StructField("branch", T.StringType(), nullable=True),  # defaults to "master"
    T.StructField("enabled", T.BooleanType(), nullable=True),  # filter flag
    T.StructField("added_at", T.TimestampType(), nullable=True),
    T.StructField("added_by", T.StringType(), nullable=True),
    T.StructField("notes", T.StringType(), nullable=True),
])


@transform_df(
    Output("ri.foundry.main.dataset.9999-adhoc-control-table"),
)
def build_control_table():
    """
    Example: hardcoded control table.
    In practice, this would be:
      - A manually maintained dataset (edited in Slate/Workshop)
      - Output of an inventory collector (e.g. your migration-orchestrator)
      - Result of a Compass folder scan
    """
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

    rows = [
        {
            "rid": "ri.foundry.main.dataset.xxxx-aaaa-yyyy-bbbb",
            "path": "/Organization/Project/bronze/dynamic_source_1",
            "branch": "master",
            "enabled": True,
            "added_at": None,
            "added_by": "migration-orchestrator",
            "notes": "Added during Phase 2 inventory scan",
        },
        {
            "rid": "ri.foundry.main.dataset.xxxx-cccc-yyyy-dddd",
            "path": "/Organization/Project/silver/dynamic_source_2",
            "branch": "master",
            "enabled": True,
            "added_at": None,
            "added_by": "manual",
            "notes": None,
        },
    ]

    return spark.createDataFrame(rows, CONTROL_SCHEMA).filter(F.col("enabled") == True)
