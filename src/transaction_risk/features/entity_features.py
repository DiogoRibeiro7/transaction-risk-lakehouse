"""Entity-level feature engineering."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from transaction_risk.validation.schema import require_columns


def add_entity_features(
    df: DataFrame,
    origin_column: str = "nameOrig",
    destination_column: str = "nameDest",
    amount_column: str = "amount",
) -> DataFrame:
    """Add account and counterparty aggregation features."""
    require_columns(df, [origin_column, destination_column, amount_column])

    origin_stats = df.groupBy(origin_column).agg(
        F.count(F.lit(1)).alias("origin_total_tx_count"),
        F.avg(amount_column).alias("origin_avg_amount"),
        F.max(amount_column).alias("origin_max_amount"),
        F.countDistinct(destination_column).alias("origin_unique_destinations"),
    )

    destination_stats = df.groupBy(destination_column).agg(
        F.count(F.lit(1)).alias("destination_total_tx_count"),
        F.avg(amount_column).alias("destination_avg_amount"),
        F.max(amount_column).alias("destination_max_amount"),
        F.countDistinct(origin_column).alias("destination_unique_origins"),
    )

    pair_stats = df.groupBy(origin_column, destination_column).agg(
        F.count(F.lit(1)).alias("origin_destination_pair_count"),
        F.avg(amount_column).alias("origin_destination_avg_amount"),
    )

    joined = (
        df.join(origin_stats, on=origin_column, how="left")
        .join(destination_stats, on=destination_column, how="left")
        .join(pair_stats, on=[origin_column, destination_column], how="left")
    )

    return joined.fillna(
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
