"""IEEE-CIS identity, card, device, and email-domain features."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

CARD_COLUMNS = [f"card{index}" for index in range(1, 7)]
EMAIL_COLUMNS = ["P_emaildomain", "R_emaildomain"]
DEVICE_COLUMNS = ["DeviceType", "DeviceInfo"]
IDENTITY_PREFIXES = ("id_", "card", "addr", "dist", "M")


def _normalize_string_column(df: DataFrame, source_column: str, target_column: str) -> DataFrame:
    """Add a lowercase normalized string feature from an optional source column."""
    if source_column not in df.columns:
        return df

    return df.withColumn(
        target_column,
        F.coalesce(F.lower(F.trim(F.col(source_column))), F.lit("unknown")),
    )


def add_identity_features(df: DataFrame) -> DataFrame:
    """Add IEEE-CIS identity and device features when relevant columns exist."""
    featured = df

    for column in CARD_COLUMNS:
        if column in featured.columns:
            featured = featured.withColumn(f"has_{column}", F.col(column).isNotNull().cast("int"))

    for column in EMAIL_COLUMNS:
        if column in featured.columns:
            normalized_column = f"{column.lower()}_normalized"
            featured = _normalize_string_column(featured, column, normalized_column).withColumn(
                f"has_{column.lower()}",
                F.col(column).isNotNull().cast("int"),
            )

    if {"P_emaildomain", "R_emaildomain"}.issubset(featured.columns):
        featured = featured.withColumn(
            "email_domains_match",
            (
                F.col("P_emaildomain").isNotNull()
                & F.col("R_emaildomain").isNotNull()
                & (
                    F.lower(F.trim(F.col("P_emaildomain")))
                    == F.lower(F.trim(F.col("R_emaildomain")))
                )
            ).cast("int"),
        )

    if "DeviceType" in featured.columns:
        featured = _normalize_string_column(featured, "DeviceType", "device_type_normalized")
    if "DeviceInfo" in featured.columns:
        featured = _normalize_string_column(featured, "DeviceInfo", "device_info_normalized")

    identity_columns = [
        column
        for column in featured.columns
        if column in DEVICE_COLUMNS
        or column in EMAIL_COLUMNS
        or any(column.startswith(prefix) for prefix in IDENTITY_PREFIXES)
    ]
    if identity_columns:
        missing_value_expression = F.lit(0)
        for column in identity_columns:
            missing_value_expression = missing_value_expression + F.col(column).isNull().cast("int")
        featured = featured.withColumn(
            "identity_missing_value_count",
            missing_value_expression,
        )

    amount_column = "TransactionAmt" if "TransactionAmt" in featured.columns else None
    if amount_column is None and "amount" in featured.columns:
        amount_column = "amount"

    if amount_column is not None and "ProductCD" in featured.columns:
        product_code_stats = featured.groupBy("ProductCD").agg(
            F.avg(F.col(amount_column)).alias("product_cd_avg_transaction_amount"),
            F.max(F.col(amount_column)).alias("product_cd_max_transaction_amount"),
            F.count(F.lit(1)).alias("product_cd_transaction_count"),
        )
        featured = featured.join(product_code_stats, on="ProductCD", how="left")

    return featured
