"""Spark-native graph-inspired transaction features.

This module does not require GraphFrames. It computes graph features with DataFrame
aggregations, which keeps the base project simple and reproducible.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from transaction_risk.validation.schema import require_columns


def add_graph_features(
    df: DataFrame,
    source_column: str = "nameOrig",
    destination_column: str = "nameDest",
    label_column: str = "isFraud",
) -> DataFrame:
    """Add graph-derived transaction network features."""
    require_columns(df, [source_column, destination_column])

    out_degree = df.groupBy(source_column).agg(F.countDistinct(destination_column).alias("origin_out_degree"))
    in_degree = df.groupBy(destination_column).agg(F.countDistinct(source_column).alias("destination_in_degree"))
    pair_frequency = df.groupBy(source_column, destination_column).agg(F.count(F.lit(1)).alias("edge_frequency"))

    result = (
        df.join(out_degree, on=source_column, how="left")
        .join(in_degree, on=destination_column, how="left")
        .join(pair_frequency, on=[source_column, destination_column], how="left")
    )

    if label_column in df.columns:
        destination_risk = df.groupBy(destination_column).agg(
            F.avg(F.col(label_column).cast("double")).alias("destination_historical_fraud_rate"),
            F.sum(F.col(label_column).cast("int")).alias("destination_historical_fraud_count"),
        )
        result = result.join(destination_risk, on=destination_column, how="left")

    fill_values = {
        "origin_out_degree": 0,
        "destination_in_degree": 0,
        "edge_frequency": 0,
        "destination_historical_fraud_rate": 0.0,
        "destination_historical_fraud_count": 0,
    }
    existing_fill_values = {key: value for key, value in fill_values.items() if key in result.columns}
    return result.fillna(existing_fill_values)
