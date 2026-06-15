"""Spark-native graph-inspired transaction features.

This module does not require GraphFrames. It computes graph features with DataFrame
aggregations, which keeps the base project simple and reproducible.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from transaction_risk.features.history import ordered_window
from transaction_risk.validation.schema import require_columns


def add_graph_features(
    df: DataFrame,
    source_column: str = "nameOrig",
    destination_column: str = "nameDest",
    label_column: str = "isFraud",
    time_column: str = "step",
) -> DataFrame:
    """Add causal graph-derived transaction network features."""
    require_columns(df, [source_column, destination_column, time_column])

    source_history = ordered_window(df, [source_column], time_column).rowsBetween(
        Window.unboundedPreceding,
        -1,
    )
    destination_history = ordered_window(df, [destination_column], time_column).rowsBetween(
        Window.unboundedPreceding,
        -1,
    )
    edge_history = ordered_window(df, [source_column, destination_column], time_column).rowsBetween(
        Window.unboundedPreceding,
        -1,
    )

    first_out_edge = F.row_number().over(
        ordered_window(df, [source_column, destination_column], time_column)
    )
    first_in_edge = F.row_number().over(
        ordered_window(df, [destination_column, source_column], time_column)
    )

    result = (
        df.withColumn(
            "_origin_new_destination",
            F.when(first_out_edge == 1, F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "_destination_new_origin",
            F.when(first_in_edge == 1, F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn("origin_out_degree", F.sum(F.col("_origin_new_destination")).over(source_history))
        .withColumn("destination_in_degree", F.sum(F.col("_destination_new_origin")).over(destination_history))
        .withColumn("edge_frequency", F.count(F.lit(1)).over(edge_history))
    )

    if label_column in df.columns:
        result = (
            result.withColumn(
                "destination_historical_fraud_count",
                F.sum(F.col(label_column).cast("int")).over(destination_history),
            )
            .withColumn(
                "_destination_history_count",
                F.count(F.lit(1)).over(destination_history),
            )
            .withColumn(
                "destination_historical_fraud_rate",
                F.when(
                    F.col("_destination_history_count") > 0,
                    F.col("destination_historical_fraud_count") / F.col("_destination_history_count"),
                ).otherwise(F.lit(0.0)),
            )
            .drop("_destination_history_count")
        )

    result = result.drop("_origin_new_destination", "_destination_new_origin")

    fill_values: dict[str, bool | float | int | str] = {
        "origin_out_degree": 0,
        "destination_in_degree": 0,
        "edge_frequency": 0,
        "destination_historical_fraud_rate": 0.0,
        "destination_historical_fraud_count": 0,
    }
    existing_fill_values = {key: value for key, value in fill_values.items() if key in result.columns}
    return result.fillna(existing_fill_values)
