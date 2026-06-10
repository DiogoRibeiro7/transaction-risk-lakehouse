"""Transaction-level feature engineering."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from transaction_risk.validation.schema import require_columns

BASE_COLUMNS = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "nameOrig",
    "nameDest",
]


def add_transaction_features(df: DataFrame) -> DataFrame:
    """Add deterministic transaction-level fraud-risk features.

    Parameters
    ----------
    df:
        Input transaction DataFrame.

    Returns
    -------
    DataFrame
        DataFrame with additional numeric and boolean risk features.
    """
    require_columns(df, BASE_COLUMNS)

    origin_delta = F.col("oldbalanceOrg") - F.col("newbalanceOrig")
    destination_delta = F.col("newbalanceDest") - F.col("oldbalanceDest")

    return (
        df.withColumn("amount_log1p", F.log1p(F.col("amount")))
        .withColumn("origin_balance_delta", origin_delta)
        .withColumn("destination_balance_delta", destination_delta)
        .withColumn("origin_delta_minus_amount", origin_delta - F.col("amount"))
        .withColumn("destination_delta_minus_amount", destination_delta - F.col("amount"))
        .withColumn("origin_balance_was_zero", (F.col("oldbalanceOrg") == 0).cast("int"))
        .withColumn("origin_balance_is_zero", (F.col("newbalanceOrig") == 0).cast("int"))
        .withColumn("destination_balance_was_zero", (F.col("oldbalanceDest") == 0).cast("int"))
        .withColumn("destination_balance_is_zero", (F.col("newbalanceDest") == 0).cast("int"))
        .withColumn("is_cash_out", (F.col("type") == F.lit("CASH_OUT")).cast("int"))
        .withColumn("is_transfer", (F.col("type") == F.lit("TRANSFER")).cast("int"))
        .withColumn("is_payment", (F.col("type") == F.lit("PAYMENT")).cast("int"))
        .withColumn("is_debit", (F.col("type") == F.lit("DEBIT")).cast("int"))
        .withColumn("is_cash_in", (F.col("type") == F.lit("CASH_IN")).cast("int"))
        .withColumn("is_merchant_destination", F.col("nameDest").startswith("M").cast("int"))
        # Keep this threshold deterministic so the same feature function works in
        # batch jobs, tests, and the local streaming demo.
        .withColumn("large_amount_flag", (F.col("amount") >= F.lit(100_000.0)).cast("int"))
    )
