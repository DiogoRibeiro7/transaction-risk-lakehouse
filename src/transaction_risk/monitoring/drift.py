"""Data drift monitoring utilities."""

from __future__ import annotations

import math

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from transaction_risk.validation.schema import require_columns


def population_stability_index(
    reference_df: DataFrame,
    current_df: DataFrame,
    column: str,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute a population stability index for a numeric column.

    This implementation collects only binned aggregate counts to the driver.
    """
    require_columns(reference_df, [column])
    require_columns(current_df, [column])
    if bins <= 1:
        raise ValueError("bins must be greater than 1.")

    quantiles = reference_df.approxQuantile(column, [i / bins for i in range(1, bins)], 0.001)
    boundaries = sorted(set(float(q) for q in quantiles))

    if not boundaries:
        return 0.0

    def with_bucket(df: DataFrame) -> DataFrame:
        bucket_expr = F.lit(0)
        for idx, boundary in enumerate(boundaries):
            bucket_expr = F.when(F.col(column) > boundary, F.lit(idx + 1)).otherwise(bucket_expr)
        return df.withColumn("bucket", bucket_expr)

    ref_counts = {row["bucket"]: row["count"] for row in with_bucket(reference_df).groupBy("bucket").count().collect()}
    cur_counts = {row["bucket"]: row["count"] for row in with_bucket(current_df).groupBy("bucket").count().collect()}
    ref_total = sum(ref_counts.values())
    cur_total = sum(cur_counts.values())

    if ref_total == 0 or cur_total == 0:
        raise ValueError("Cannot compute PSI on empty reference or current data.")

    psi = 0.0
    for bucket in range(len(boundaries) + 1):
        expected = max(ref_counts.get(bucket, 0) / ref_total, epsilon)
        observed = max(cur_counts.get(bucket, 0) / cur_total, epsilon)
        psi += (observed - expected) * math.log(observed / expected)
    return float(psi)


def drift_report(
    reference_df: DataFrame,
    current_df: DataFrame,
    columns: list[str],
    bins: int = 10,
) -> dict[str, float]:
    """Compute PSI for several numeric columns."""
    if not columns:
        raise ValueError("columns must not be empty.")
    return {
        column: population_stability_index(reference_df, current_df, column=column, bins=bins)
        for column in columns
    }


def score_distribution_drift(
    reference_df: DataFrame,
    current_df: DataFrame,
    score_column: str = "fraud_probability",
    bins: int = 10,
) -> float:
    """Compute PSI between reference and current model score distributions."""
    return population_stability_index(
        reference_df,
        current_df,
        column=score_column,
        bins=bins,
    )


def label_rate_drift(
    reference_df: DataFrame,
    current_df: DataFrame,
    label_column: str = "isFraud",
) -> dict[str, float]:
    """Compare positive label rates between reference and current windows."""
    require_columns(reference_df, [label_column])
    require_columns(current_df, [label_column])

    def positive_rate(df: DataFrame) -> float:
        row = df.agg(F.avg(F.col(label_column).cast("double")).alias("rate")).collect()[0]
        return float(row["rate"] or 0.0)

    reference_rate = positive_rate(reference_df)
    current_rate = positive_rate(current_df)
    absolute_change = current_rate - reference_rate
    relative_change = absolute_change / reference_rate if reference_rate else 0.0

    return {
        "reference_label_rate": reference_rate,
        "current_label_rate": current_rate,
        "absolute_change": absolute_change,
        "relative_change": relative_change,
    }
