"""Temporal feature engineering for ordered transactions."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from transaction_risk.validation.schema import require_columns


def add_temporal_features(
    df: DataFrame,
    entity_column: str = "nameOrig",
    time_column: str = "step",
    amount_column: str = "amount",
) -> DataFrame:
    """Add temporal account behaviour features.

    The PaySim `step` column is an ordered time index. This function uses Spark windows
    to compute previous transaction timing and rolling history features.
    """
    require_columns(df, [entity_column, time_column, amount_column])

    by_entity_time = Window.partitionBy(entity_column).orderBy(time_column)
    history_window = by_entity_time.rowsBetween(Window.unboundedPreceding, -1)

    return (
        df.withColumn("previous_step_by_origin", F.lag(F.col(time_column)).over(by_entity_time))
        .withColumn(
            "steps_since_previous_origin_tx",
            F.coalesce(F.col(time_column) - F.col("previous_step_by_origin"), F.lit(-1)),
        )
        .withColumn("origin_tx_count_before", F.count(F.lit(1)).over(history_window))
        .withColumn("origin_amount_mean_before", F.avg(F.col(amount_column)).over(history_window))
        .withColumn("origin_amount_std_before", F.stddev_pop(F.col(amount_column)).over(history_window))
        .withColumn(
            "origin_amount_to_mean_ratio",
            F.when(F.col("origin_amount_mean_before") > 0, F.col(amount_column) / F.col("origin_amount_mean_before"))
            .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "origin_amount_zscore_before",
            F.when(
                F.col("origin_amount_std_before") > 0,
                (F.col(amount_column) - F.col("origin_amount_mean_before")) / F.col("origin_amount_std_before"),
            ).otherwise(F.lit(0.0)),
        )
        .fillna(
            {
                "origin_tx_count_before": 0,
                "origin_amount_mean_before": 0.0,
                "origin_amount_std_before": 0.0,
                "origin_amount_to_mean_ratio": 0.0,
                "origin_amount_zscore_before": 0.0,
            }
        )
    )
