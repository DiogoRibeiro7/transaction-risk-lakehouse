"""Threshold selection utilities."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from transaction_risk.validation.schema import require_columns


def threshold_for_alert_rate(
    scored_df: DataFrame,
    alert_rate: float,
    probability_column: str = "fraud_probability",
) -> float:
    """Select a probability threshold that approximately produces a target alert rate."""
    require_columns(scored_df, [probability_column])
    if not 0 < alert_rate < 1:
        raise ValueError("alert_rate must be between 0 and 1.")

    quantile = 1.0 - alert_rate
    threshold = scored_df.approxQuantile(probability_column, [quantile], 0.001)[0]
    return float(threshold)


def add_alert_flag(
    scored_df: DataFrame,
    threshold: float,
    probability_column: str = "fraud_probability",
    output_column: str = "is_alert",
) -> DataFrame:
    """Add an alert flag based on a fraud probability threshold."""
    require_columns(scored_df, [probability_column])
    return scored_df.withColumn(output_column, (F.col(probability_column) >= threshold).cast("int"))


def add_top_k_alert_flag(
    scored_df: DataFrame,
    k: int,
    probability_column: str = "fraud_probability",
    output_column: str = "is_top_k_alert",
) -> DataFrame:
    """Flag the top-k highest-risk transactions."""
    require_columns(scored_df, [probability_column])
    if k <= 0:
        raise ValueError("k must be positive.")

    window = Window.orderBy(F.col(probability_column).desc())
    return (
        scored_df.withColumn("risk_rank", F.row_number().over(window))
        .withColumn(output_column, (F.col("risk_rank") <= k).cast("int"))
    )
