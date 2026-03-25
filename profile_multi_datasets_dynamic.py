"""
profile_multi_datasets_dynamic.py
──────────────────────────────────
Foundry Code Repository transform that dynamically reads dataset RIDs at
runtime from a driver dataset, profiles each using foundry-profiler v2.1
(with inline fallback), and writes a unified profiling output.

Key design:
  - Uses ctx._foundry.input(rid=..., branch=...) for runtime dataset reads.
  - Uses ctx.spark_session for all DataFrame creation (profile rows, empties).
  - The driver dataset (rids_registry) is the ONLY declared input - all other
    datasets are resolved dynamically.

GOVERNANCE TRADE-OFFS (accepted by design):
  - No lineage tracking for dynamically read datasets.
  - No automatic markings propagation from dynamic inputs to output.
  - No transaction isolation - each dynamic read hits the latest committed
    transaction, not a pinned snapshot.
  - Requires Project Scope Exemption enabled in the enrollment.
  - ctx._foundry is an undocumented internal API and may change across
    platform upgrades.

Driver dataset schema (rids_registry):
  | rid (string)                                | dataset_path (string)                   | branch |
  | ri.foundry.main.dataset.aaaa-1111-bbbb-2222 | /Organization/Project/bronze/source_a   | master |
  | ri.foundry.main.dataset.cccc-3333-dddd-4444 | /Organization/Project/silver/dataset_x  | master |
"""

import logging
from functools import reduce
from datetime import datetime

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T
from transforms.api import transform, Input, Output, TransformContext

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 1. PROFILING OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────────

PROFILE_SCHEMA = T.StructType([
    T.StructField("column_name", T.StringType()),
    T.StructField("data_type", T.StringType()),
    T.StructField("row_count", T.LongType()),
    T.StructField("null_count", T.LongType()),
    T.StructField("null_pct", T.DoubleType()),
    T.StructField("distinct_count", T.LongType()),
    T.StructField("uniqueness_ratio", T.DoubleType()),
    T.StructField("min_value", T.StringType()),
    T.StructField("max_value", T.StringType()),
    T.StructField("mean", T.DoubleType()),
    T.StructField("stddev_pop", T.DoubleType()),
    T.StructField("cv", T.DoubleType()),
    T.StructField("numeric_mode", T.StringType()),
    T.StructField("min_length", T.IntegerType()),
    T.StructField("max_length", T.IntegerType()),
    T.StructField("avg_length", T.DoubleType()),
    T.StructField("source_rid", T.StringType()),
    T.StructField("dataset_path", T.StringType()),
    T.StructField("profiled_at", T.TimestampType()),
])

NUMERIC_TYPES = frozenset({
    "int", "bigint", "smallint", "tinyint",
    "float", "double", "decimal", "long", "short",
})


# ─────────────────────────────────────────────────────────────
# 2. PROFILER — library import with inline fallback
# ─────────────────────────────────────────────────────────────

try:
    from foundry_profiler import profile_dataframe as _lib_profile

    def profile_dataframe(spark, df, source_rid, dataset_path):
        """Delegate to foundry-profiler v2.1 library, tag with source metadata."""
        result = _lib_profile(df)
        return (
            result
            .withColumn("source_rid", F.lit(source_rid))
            .withColumn("dataset_path", F.lit(dataset_path))
        )

    logger.info("Using foundry-profiler library for profiling.")

except ImportError:
    logger.warning(
        "foundry-profiler not available; using inline profiler."
    )

    def profile_dataframe(spark, df, source_rid, dataset_path):
        """
        Self-contained inline profiler matching foundry-profiler v2.1 metrics.

        Uses ctx.spark_session (passed as `spark`) for all DataFrame creation.
        Single-pass batched aggregation for core metrics, separate GroupBy for
        numeric mode.

        Args:
            spark: SparkSession from ctx.spark_session
            df: Input pyspark.sql.DataFrame to profile
            source_rid: Dataset RID string for tagging
            dataset_path: Dataset path string for tagging

        Returns:
            DataFrame with one row per column, matching PROFILE_SCHEMA
        """
        columns = df.columns
        dtypes = dict(df.dtypes)
        row_count_val = df.count()

        if row_count_val == 0:
            return spark.createDataFrame([], PROFILE_SCHEMA)

        # ── Batched aggregation: single pass over all columns ──
        agg_exprs = []
        for col_name in columns:
            c = F.col("`{}`".format(col_name))
            col_type = dtypes[col_name]
            is_numeric = col_type in NUMERIC_TYPES
            is_string = col_type == "string"

            agg_exprs.extend([
                F.sum(F.when(c.isNull(), 1).otherwise(0))
                    .alias("{0}__null_count".format(col_name)),
                F.countDistinct(c)
                    .alias("{0}__distinct_count".format(col_name)),
            ])

            if is_numeric:
                agg_exprs.extend([
                    F.min(c).alias("{0}__min".format(col_name)),
                    F.max(c).alias("{0}__max".format(col_name)),
                    F.mean(c.cast("double")).alias("{0}__mean".format(col_name)),
                    F.stddev_pop(c.cast("double")).alias("{0}__stddev_pop".format(col_name)),
                ])
            elif is_string:
                agg_exprs.extend([
                    F.min(c).alias("{0}__min".format(col_name)),
                    F.max(c).alias("{0}__max".format(col_name)),
                    F.min(F.length(c)).alias("{0}__min_length".format(col_name)),
                    F.max(F.length(c)).alias("{0}__max_length".format(col_name)),
                    F.mean(F.length(c).cast("double")).alias("{0}__avg_length".format(col_name)),
                ])
            else:
                agg_exprs.extend([
                    F.min(c.cast("string")).alias("{0}__min".format(col_name)),
                    F.max(c.cast("string")).alias("{0}__max".format(col_name)),
                ])

        agg_row = df.agg(*agg_exprs).collect()[0]

        # ── Numeric mode via GroupBy ──
        numeric_modes = {}
        for col_name in columns:
            if dtypes[col_name] not in NUMERIC_TYPES:
                continue
            try:
                mode_row = (
                    df.filter(F.col("`{}`".format(col_name)).isNotNull())
                    .groupBy(F.col("`{}`".format(col_name)))
                    .count()
                    .orderBy(F.desc("count"))
                    .limit(1)
                    .collect()
                )
                numeric_modes[col_name] = (
                    str(mode_row[0][col_name]) if mode_row else None
                )
            except Exception:
                numeric_modes[col_name] = None

        # ── Assemble profile rows ──
        profiled_at = datetime.utcnow()
        rows = []

        for col_name in columns:
            col_type = dtypes[col_name]
            is_numeric = col_type in NUMERIC_TYPES
            is_string = col_type == "string"

            null_count = int(agg_row["{0}__null_count".format(col_name)] or 0)
            distinct_count = int(agg_row["{0}__distinct_count".format(col_name)] or 0)
            non_null_count = row_count_val - null_count
            null_pct = (null_count / row_count_val) if row_count_val > 0 else 0.0
            uniqueness_ratio = (
                (distinct_count / non_null_count) if non_null_count > 0 else 0.0
            )

            min_val = max_val = None
            mean_val = stddev_val = cv_val = mode_val = None
            min_len = max_len = avg_len = None

            if is_numeric:
                raw_min = agg_row["{0}__min".format(col_name)]
                raw_max = agg_row["{0}__max".format(col_name)]
                min_val = str(raw_min) if raw_min is not None else None
                max_val = str(raw_max) if raw_max is not None else None
                mean_val = agg_row["{0}__mean".format(col_name)]
                stddev_val = agg_row["{0}__stddev_pop".format(col_name)]
                if mean_val and mean_val != 0 and stddev_val is not None:
                    cv_val = stddev_val / mean_val
                mode_val = numeric_modes.get(col_name)
            elif is_string:
                min_val = agg_row.asDict().get("{0}__min".format(col_name))
                max_val = agg_row.asDict().get("{0}__max".format(col_name))
                min_len = agg_row.asDict().get("{0}__min_length".format(col_name))
                max_len = agg_row.asDict().get("{0}__max_length".format(col_name))
                avg_len = agg_row.asDict().get("{0}__avg_length".format(col_name))
            else:
                min_val = agg_row.asDict().get("{0}__min".format(col_name))
                max_val = agg_row.asDict().get("{0}__max".format(col_name))

            rows.append((
                col_name, col_type, row_count_val,
                null_count,
                round(null_pct, 6),
                distinct_count,
                round(uniqueness_ratio, 6),
                min_val, max_val,
                round(mean_val, 6) if mean_val is not None else None,
                round(stddev_val, 6) if stddev_val is not None else None,
                round(cv_val, 6) if cv_val is not None else None,
                mode_val,
                min_len, max_len,
                round(avg_len, 4) if avg_len is not None else None,
                source_rid, dataset_path, profiled_at,
            ))

        return spark.createDataFrame(rows, PROFILE_SCHEMA)


# ─────────────────────────────────────────────────────────────
# 3. TRANSFORM — dynamic runtime profiling via ctx
# ─────────────────────────────────────────────────────────────

@transform(
    rids_registry=Input("ri.foundry.main.dataset.YOUR_RIDS_REGISTRY_RID"),
    output=Output("ri.foundry.main.dataset.YOUR_PROFILING_OUTPUT_RID"),
)
def profile_datasets_dynamic(ctx, rids_registry, output):
    # type: (TransformContext, ..., ...) -> None
    """
    Dynamically reads datasets from RIDs listed in the rids_registry driver
    dataset at runtime, profiles each, and writes a unified output.

    Flow:
      1. Read rids_registry -> collect list of (rid, dataset_path, branch)
      2. For each entry, use ctx._foundry.input(rid=..., branch=...) to
         read the dataset as a TransformInput, then .dataframe()
      3. Profile using foundry-profiler v2.1 (or inline fallback),
         passing ctx.spark_session for DataFrame creation
      4. Union all profiles -> write to output
    """
    spark = ctx.spark_session

    # ── Step 1: Read driver dataset ──
    registry_df = rids_registry.dataframe()
    entries = registry_df.select("rid", "dataset_path", "branch").collect()

    if not entries:
        logger.warning("rids_registry is empty. Aborting.")
        output.abort()
        return

    logger.info("Found {0} datasets to profile in rids_registry.".format(len(entries)))

    # ── Step 2 & 3: Loop, read dynamically, profile ──
    profiles = []
    failed = []

    for entry in entries:
        rid = entry["rid"]
        dataset_path = entry["dataset_path"]
        branch = entry["branch"] or "master"

        logger.info("Reading dataset: {0} ({1}) @ {2}".format(dataset_path, rid, branch))

        try:
            # ═══════════════════════════════════════════════════════
            # DYNAMIC READ via ctx._foundry.input()
            # This is the key undocumented API call that resolves
            # a dataset RID at runtime (not CI time).
            # ═══════════════════════════════════════════════════════
            dynamic_input = ctx._foundry.input(rid=rid, branch=branch)
            df = dynamic_input.dataframe()

            logger.info(
                "  -> Schema: {0} columns, profiling...".format(len(df.columns))
            )

            # Profile using ctx.spark_session for DataFrame creation
            profile = profile_dataframe(
                spark=spark,
                df=df,
                source_rid=rid,
                dataset_path=dataset_path,
            )
            profiles.append(profile)

            logger.info("  OK Profiled {0}".format(dataset_path))

        except Exception as e:
            logger.error("  FAIL {0} ({1}): {2}".format(dataset_path, rid, e))
            failed.append({"rid": rid, "path": dataset_path, "error": str(e)})
            continue

    # ── Step 4: Union and write ──
    if not profiles:
        logger.error(
            "All {0} datasets failed profiling. Aborting build.".format(len(entries))
        )
        output.abort()
        return

    unioned = reduce(
        lambda a, b: a.unionByName(b, allowMissingColumns=True),
        profiles[1:],
        profiles[0],
    )

    # Add build-level metadata
    unioned = (
        unioned
        .withColumn("build_timestamp", F.current_timestamp())
        .withColumn("total_datasets_profiled", F.lit(len(profiles)))
        .withColumn("total_datasets_failed", F.lit(len(failed)))
    )

    output.write_dataframe(unioned)

    logger.info(
        "Done. Profiled {0}/{1} datasets ({2} failed). Output rows: pending.".format(
            len(profiles), len(entries), len(failed)
        )
    )

    if failed:
        logger.warning("Failed datasets:")
        for f_entry in failed:
            logger.warning("  - {0}: {1}".format(f_entry["path"], f_entry["error"]))
