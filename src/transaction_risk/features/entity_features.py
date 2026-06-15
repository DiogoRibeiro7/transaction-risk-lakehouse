"""Entity-level feature engineering."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from transaction_risk.features.history import ordered_window
from transaction_risk.validation.schema import require_columns


def add_entity_features(
    df: DataFrame,
    origin_column: str = "nameOrig",
    destination_column: str = "nameDest",
    amount_column: str = "amount",
    time_column: str = "step",
) -> DataFrame:
    """Add causal account and counterparty aggregation features."""
    require_columns(df, [origin_column, destination_column, amount_column, time_column])

    origin_history = ordered_window(df, [origin_column], time_column).rowsBetween(
        Window.unboundedPreceding,
        -1,
    )
    destination_history = ordered_window(df, [destination_column], time_column).rowsBetween(
        Window.unboundedPreceding,
        -1,
    )
    pair_history = ordered_window(df, [origin_column, destination_column], time_column).rowsBetween(
        Window.unboundedPreceding,
        -1,
    )

    first_origin_destination = F.row_number().over(
        ordered_window(df, [origin_column, destination_column], time_column)
    )
    first_destination_origin = F.row_number().over(
        ordered_window(df, [destination_column, origin_column], time_column)
    )

    featured = (
        df.withColumn("origin_total_tx_count", F.count(F.lit(1)).over(origin_history))
        .withColumn("origin_avg_amount", F.avg(F.col(amount_column)).over(origin_history))
        .withColumn("origin_max_amount", F.max(F.col(amount_column)).over(origin_history))
        .withColumn("destination_total_tx_count", F.count(F.lit(1)).over(destination_history))
        .withColumn("destination_avg_amount", F.avg(F.col(amount_column)).over(destination_history))
        .withColumn("destination_max_amount", F.max(F.col(amount_column)).over(destination_history))
        .withColumn("origin_destination_pair_count", F.count(F.lit(1)).over(pair_history))
        .withColumn("origin_destination_avg_amount", F.avg(F.col(amount_column)).over(pair_history))
        .withColumn(
            "_origin_new_destination",
            F.when(first_origin_destination == 1, F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "_destination_new_origin",
            F.when(first_destination_origin == 1, F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "origin_unique_destinations",
            F.sum(F.col("_origin_new_destination")).over(origin_history),
        )
        .withColumn(
            "destination_unique_origins",
            F.sum(F.col("_destination_new_origin")).over(destination_history),
        )
        .drop("_origin_new_destination", "_destination_new_origin")
    )

    return featured.fillna(
        {
            "origin_total_tx_count": 0,
            "origin_avg_amount": 0.0,
            "origin_max_amount": 0.0,
            "origin_unique_destinations": 0,
            "destination_total_tx_count": 0,
            "destination_avg_amount": 0.0,
            "destination_max_amount": 0.0,
            "destination_unique_origins": 0,
            "origin_destination_pair_count": 0,
            "origin_destination_avg_amount": 0.0,
        }
    )
