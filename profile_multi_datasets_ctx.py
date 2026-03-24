"""
profile_multi_datasets_ctx.py
──────────────────────────────
Dual-mode Foundry profiler transform using TransformContext:

  MODE A — Governed (semi-dynamic):
    RIDs declared at CI time via rid_config.json → full lineage + markings.

  MODE B — Ad-hoc (dynamic runtime):
    RIDs read from a control dataset at runtime via ctx._foundry.input(rid).
    ⚠ No lineage, no markings propagation, no transaction isolation.
    Requires Project Scope Exemption.

Both modes:
  - Use ctx.spark_session for empty DF creation and fallback schemas
  - Use ctx.auth_header to call Foundry Metadata API for dataset enrichment
  - Profile via foundry-profiler v2.1 (with inline fallback)
  - Tag results with source_rid + dataset_path + API metadata
  - Union all profiles into a single output
"""

import json
import logging
import requests
from functools import reduce
from pathlib import Path
from datetime import datetime

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T
from transforms.api import transform, Input, Output, TransformContext

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent / "rid_config.json"

with open(_CONFIG_PATH, "r") as _f:
    _CONFIG = json.load(_f)

# Expected schema — see rid_config.json
_GOVERNED_DATASETS = _CONFIG.get("governed_datasets", [])
_OUTPUT_RID = _CONFIG["output_rid"]
_FOUNDRY_HOSTNAME = _CONFIG.get("hostname", "")  # e.g. "myorg.palantirfoundry.com"

# ─────────────────────────────────────────────────────────────
# 2. PROFILE OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────────

PROFILE_SCHEMA = T.StructType([
    # Column-level metrics
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
    # Source tagging
    T.StructField("source_rid", T.StringType()),
    T.StructField("dataset_path", T.StringType()),
    T.StructField("profiling_mode", T.StringType()),  # "governed" or "ad_hoc"
    # API-enriched metadata
    T.StructField("dataset_name", T.StringType()),
    T.StructField("last_modified", T.StringType()),
    T.StructField("last_transaction_rid", T.StringType()),
    T.StructField("schema_branch", T.StringType()),
    # Build metadata
    T.StructField("profiled_at", T.TimestampType()),
    T.StructField("build_timestamp", T.TimestampType()),
    T.StructField("total_datasets_profiled", T.IntegerType()),
])

# ─────────────────────────────────────────────────────────────
# 3. METADATA ENRICHMENT via ctx.auth_header
# ─────────────────────────────────────────────────────────────

def fetch_dataset_metadata(auth_header, hostname, dataset_rid, branch="master"):
    """
    Call Foundry Metadata/Catalog APIs to retrieve dataset-level metadata.

    Uses two endpoints:
      - GET /api/v2/datasets/{rid}          → dataset name, path
      - GET /api/v2/datasets/{rid}/branches/{branch}  → last transaction

    Returns a dict with enrichment fields; empty values on failure.
    """
    meta = {
        "dataset_name": None,
        "last_modified": None,
        "last_transaction_rid": None,
        "schema_branch": branch,
    }

    headers = {"Authorization": auth_header}
    base_url = f"https://{hostname}"

    # ── Dataset metadata ──
    try:
        resp = requests.get(
            f"{base_url}/api/v2/datasets/{dataset_rid}",
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            meta["dataset_name"] = data.get("name")
            meta["last_modified"] = data.get("lastModified")
        else:
            logger.warning(
                f"Dataset API returned {resp.status_code} for {dataset_rid}"
            )
    except Exception as e:
        logger.warning(f"Failed to fetch dataset metadata for {dataset_rid}: {e}")

    # ── Branch / last transaction ──
    try:
        resp = requests.get(
            f"{base_url}/api/v2/datasets/{dataset_rid}/branches/{branch}",
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            meta["last_transaction_rid"] = data.get("transactionRid")
        else:
            logger.warning(
                f"Branch API returned {resp.status_code} for {dataset_rid}/{branch}"
            )
    except Exception as e:
        logger.warning(f"Failed to fetch branch metadata for {dataset_rid}: {e}")

    return meta


# ─────────────────────────────────────────────────────────────
# 4. PROFILER — library import with inline fallback
# ─────────────────────────────────────────────────────────────

try:
    from foundry_profiler import profile_dataframe as _lib_profile

    def _run_profiler(df, spark):
        """Delegate to foundry-profiler library."""
        return _lib_profile(df)

    logger.info("Using foundry-profiler library for profiling.")

except ImportError:
    logger.warning(
        "foundry-profiler not available; falling back to inline profiler."
    )

    _NUMERIC_TYPES = frozenset({
        "int", "bigint", "smallint", "tinyint",
        "float", "double", "decimal", "long", "short",
    })

    def _run_profiler(df, spark):
        """
        Self-contained inline profiler matching foundry-profiler v2.1.
        Single-pass batched aggregation + parallel GroupBy for numeric mode.
        """
        columns = df.columns
        dtypes = dict(df.dtypes)
        row_count_val = df.count()

        if row_count_val == 0:
            return spark.createDataFrame([], PROFILE_SCHEMA)

        # ── Batched aggregation: one select() for all columns ──
        agg_exprs = []
        for col_name in columns:
            c = F.col(f"`{col_name}`")
            agg_exprs.extend([
                F.sum(F.when(c.isNull(), 1).otherwise(0)).alias(f"{col_name}__null_count"),
                F.countDistinct(c).alias(f"{col_name}__distinct_count"),
            ])

            col_type = dtypes[col_name]
            is_numeric = col_type in _NUMERIC_TYPES
            is_string = col_type == "string"

            if is_numeric:
                agg_exprs.extend([
                    F.min(c).alias(f"{col_name}__min"),
                    F.max(c).alias(f"{col_name}__max"),
                    F.mean(c.cast("double")).alias(f"{col_name}__mean"),
                    F.stddev_pop(c.cast("double")).alias(f"{col_name}__stddev_pop"),
                ])
            elif is_string:
                agg_exprs.extend([
                    F.min(c).alias(f"{col_name}__min"),
                    F.max(c).alias(f"{col_name}__max"),
                    F.min(F.length(c)).alias(f"{col_name}__min_length"),
                    F.max(F.length(c)).alias(f"{col_name}__max_length"),
                    F.mean(F.length(c).cast("double")).alias(f"{col_name}__avg_length"),
                ])
            else:
                agg_exprs.extend([
                    F.min(c.cast("string")).alias(f"{col_name}__min"),
                    F.max(c.cast("string")).alias(f"{col_name}__max"),
                ])

        agg_row = df.agg(*agg_exprs).collect()[0]

        # ── Numeric mode via GroupBy ──
        numeric_modes = {}
        for col_name in columns:
            if dtypes[col_name] in _NUMERIC_TYPES:
                try:
                    mode_row = (
                        df.filter(F.col(f"`{col_name}`").isNotNull())
                        .groupBy(F.col(f"`{col_name}`"))
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

        # ── Assemble rows ──
        profiled_at = datetime.utcnow()
        rows = []

        for col_name in columns:
            col_type = dtypes[col_name]
            is_numeric = col_type in _NUMERIC_TYPES
            is_string = col_type == "string"

            null_count = int(agg_row[f"{col_name}__null_count"] or 0)
            distinct_count = int(agg_row[f"{col_name}__distinct_count"] or 0)
            non_null = row_count_val - null_count
            null_pct = (null_count / row_count_val) if row_count_val > 0 else 0.0
            uniqueness_ratio = (distinct_count / non_null) if non_null > 0 else 0.0

            row_data = {
                "column_name": col_name,
                "data_type": col_type,
                "row_count": row_count_val,
                "null_count": null_count,
                "null_pct": round(null_pct, 6),
                "distinct_count": distinct_count,
                "uniqueness_ratio": round(uniqueness_ratio, 6),
                "min_value": None,
                "max_value": None,
                "mean": None,
                "stddev_pop": None,
                "cv": None,
                "numeric_mode": None,
                "min_length": None,
                "max_length": None,
                "avg_length": None,
            }

            if is_numeric:
                raw_min = agg_row[f"{col_name}__min"]
                raw_max = agg_row[f"{col_name}__max"]
                row_data["min_value"] = str(raw_min) if raw_min is not None else None
                row_data["max_value"] = str(raw_max) if raw_max is not None else None
                mean_val = agg_row[f"{col_name}__mean"]
                stddev_val = agg_row[f"{col_name}__stddev_pop"]
                row_data["mean"] = round(mean_val, 6) if mean_val is not None else None
                row_data["stddev_pop"] = round(stddev_val, 6) if stddev_val is not None else None
                row_data["cv"] = (
                    round(stddev_val / mean_val, 6)
                    if mean_val and mean_val != 0 and stddev_val is not None
                    else None
                )
                row_data["numeric_mode"] = numeric_modes.get(col_name)
            elif is_string:
                row_data["min_value"] = agg_row.get(f"{col_name}__min")
                row_data["max_value"] = agg_row.get(f"{col_name}__max")
                row_data["min_length"] = agg_row.get(f"{col_name}__min_length")
                row_data["max_length"] = agg_row.get(f"{col_name}__max_length")
                avg_len = agg_row.get(f"{col_name}__avg_length")
                row_data["avg_length"] = round(avg_len, 4) if avg_len is not None else None
            else:
                row_data["min_value"] = agg_row.get(f"{col_name}__min")
                row_data["max_value"] = agg_row.get(f"{col_name}__max")

            rows.append(row_data)

        return spark.createDataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 5. CORE PROFILING ENGINE — shared by both modes
# ─────────────────────────────────────────────────────────────

def profile_and_tag(df, spark, source_rid, dataset_path, mode, api_meta=None):
    """
    Profile a single DataFrame and tag with source + API metadata.

    Args:
        df:            Spark DataFrame to profile
        spark:         SparkSession from ctx.spark_session
        source_rid:    Dataset RID string
        dataset_path:  Dataset path string
        mode:          "governed" or "ad_hoc"
        api_meta:      Dict from fetch_dataset_metadata() or None

    Returns:
        Spark DataFrame with profile rows + metadata columns
    """
    api_meta = api_meta or {}

    profile = _run_profiler(df, spark)

    return (
        profile
        .withColumn("source_rid", F.lit(source_rid))
        .withColumn("dataset_path", F.lit(dataset_path))
        .withColumn("profiling_mode", F.lit(mode))
        .withColumn("dataset_name", F.lit(api_meta.get("dataset_name")))
        .withColumn("last_modified", F.lit(api_meta.get("last_modified")))
        .withColumn("last_transaction_rid", F.lit(api_meta.get("last_transaction_rid")))
        .withColumn("schema_branch", F.lit(api_meta.get("schema_branch", "master")))
        .withColumn("profiled_at", F.lit(datetime.utcnow()).cast(T.TimestampType()))
    )


# ─────────────────────────────────────────────────────────────
# 6. MODE A — Governed: CI-time declared inputs
# ─────────────────────────────────────────────────────────────

def _build_governed_transform(governed_datasets, output_rid, hostname):
    """
    Factory: generates a @transform with statically declared inputs.
    Full lineage, markings propagation, transaction isolation.
    """

    input_kwargs = {
        f"gov_{i}": Input(entry["rid"])
        for i, entry in enumerate(governed_datasets)
    }

    key_to_meta = {
        f"gov_{i}": {"rid": entry["rid"], "path": entry["path"]}
        for i, entry in enumerate(governed_datasets)
    }

    @transform(
        output=Output(output_rid),
        **input_kwargs,
    )
    def governed_profiler(ctx: TransformContext, output, **kwargs):
        spark = ctx.spark_session
        auth = ctx.auth_header

        profiles = []

        for key, transform_input in kwargs.items():
            meta = key_to_meta[key]
            rid = meta["rid"]
            path = meta["path"]
            logger.info(f"[GOVERNED] Profiling: {path} ({rid})")

            try:
                df = transform_input.dataframe()

                # Enrich via API
                api_meta = fetch_dataset_metadata(auth, hostname, rid)

                profile = profile_and_tag(
                    df, spark,
                    source_rid=rid,
                    dataset_path=path,
                    mode="governed",
                    api_meta=api_meta,
                )
                profiles.append(profile)
                logger.info(f"  ✓ Profiled {len(df.columns)} columns")
            except Exception as e:
                logger.error(f"  ✗ Failed: {path} ({rid}): {e}")
                continue

        _write_profiles(profiles, output, spark)

    return governed_profiler


# ─────────────────────────────────────────────────────────────
# 7. MODE B — Ad-hoc: dynamic runtime reads via ctx._foundry
# ─────────────────────────────────────────────────────────────

def _build_adhoc_transform(output_rid, hostname):
    """
    Factory: generates a @transform that reads a control dataset containing
    RIDs, then dynamically reads each via ctx._foundry.input(rid).

    ⚠ Requires Project Scope Exemption.
    ⚠ No lineage / markings / transaction isolation for dynamic reads.
    """

    @transform(
        output=Output(output_rid),
        control=Input(_CONFIG["adhoc_control_rid"]),
    )
    def adhoc_profiler(ctx: TransformContext, output, control):
        spark = ctx.spark_session
        auth = ctx.auth_header

        # Read RID list from control dataset
        # Expected schema: rid (string), path (string), branch (string, optional)
        control_df = control.dataframe()
        rid_rows = control_df.select("rid", "path").collect()

        if not rid_rows:
            logger.warning("[AD-HOC] Control dataset is empty. Aborting.")
            output.abort()
            return

        logger.info(f"[AD-HOC] Found {len(rid_rows)} datasets to profile dynamically.")

        profiles = []

        for row in rid_rows:
            rid = row["rid"]
            path = row["path"]
            branch = row["branch"] if "branch" in control_df.columns else "master"

            logger.info(f"[AD-HOC] Profiling: {path} ({rid}) on branch={branch}")

            try:
                # ── Dynamic read via ctx._foundry.input() ──
                dynamic_input = ctx._foundry.input(rid=rid, branch=branch)
                df = dynamic_input.dataframe()

                # Enrich via API
                api_meta = fetch_dataset_metadata(auth, hostname, rid, branch)

                profile = profile_and_tag(
                    df, spark,
                    source_rid=rid,
                    dataset_path=path,
                    mode="ad_hoc",
                    api_meta=api_meta,
                )
                profiles.append(profile)
                logger.info(f"  ✓ Profiled {len(df.columns)} columns")
            except Exception as e:
                logger.error(f"  ✗ Failed: {path} ({rid}): {e}")
                continue

        _write_profiles(profiles, output, spark)

    return adhoc_profiler


# ─────────────────────────────────────────────────────────────
# 8. SHARED OUTPUT WRITER
# ─────────────────────────────────────────────────────────────

def _write_profiles(profiles, output, spark):
    """Union all profile DataFrames and write to output, or abort if empty."""
    if not profiles:
        logger.warning("No datasets were successfully profiled. Aborting output.")
        output.abort()
        return

    unioned = reduce(
        lambda a, b: a.unionByName(b, allowMissingColumns=True),
        profiles[1:],
        profiles[0],
    )

    unioned = (
        unioned
        .withColumn("build_timestamp", F.current_timestamp())
        .withColumn("total_datasets_profiled", F.lit(len(profiles)))
    )

    output.write_dataframe(unioned)
    logger.info(
        f"Wrote unified profile: {unioned.count()} rows "
        f"from {len(profiles)} datasets."
    )


# ─────────────────────────────────────────────────────────────
# 9. REGISTER TRANSFORMS — Foundry discovers at CI
# ─────────────────────────────────────────────────────────────

TRANSFORMS = []

# Always register governed transform if governed_datasets exist
if _GOVERNED_DATASETS:
    TRANSFORMS.append(
        _build_governed_transform(_GOVERNED_DATASETS, _OUTPUT_RID, _FOUNDRY_HOSTNAME)
    )

# Register ad-hoc transform only if control RID is configured
if _CONFIG.get("adhoc_control_rid"):
    _ADHOC_OUTPUT = _CONFIG.get("adhoc_output_rid", _OUTPUT_RID)
    TRANSFORMS.append(
        _build_adhoc_transform(_ADHOC_OUTPUT, _FOUNDRY_HOSTNAME)
    )
