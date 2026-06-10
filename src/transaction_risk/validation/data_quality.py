"""Data quality checks for transaction tables."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from transaction_risk.validation.schema import require_columns


def null_count_report(df: DataFrame, columns: list[str] | None = None) -> DataFrame:
    """Return a one-row DataFrame with null counts for selected columns."""
    selected_columns = columns or df.columns
    require_columns(df, selected_columns)
    expressions = [F.sum(F.col(column).isNull().cast("int")).alias(column) for column in selected_columns]
    return df.select(*expressions)


def duplicate_count(df: DataFrame, key_columns: list[str]) -> int:
    """Count duplicated keys in a DataFrame."""
    require_columns(df, key_columns)
    duplicate_rows = df.groupBy(*key_columns).count().filter(F.col("count") > 1)
    return int(duplicate_rows.count())


def class_balance_report(df: DataFrame, label_column: str = "isFraud") -> DataFrame:
    """Return class counts and class proportions for a binary label."""
    require_columns(df, [label_column])
    total = df.count()
    if total == 0:
        raise ValueError("Cannot compute class balance for an empty DataFrame.")

    return (
        df.groupBy(label_column)
        .count()
        .withColumn("proportion", F.col("count") / F.lit(total))
        .orderBy(label_column)
    )


def basic_quality_report(df: DataFrame, label_column: str = "isFraud") -> dict[str, int | float]:
    """Compute small scalar quality metrics suitable for logging or JSON export."""
    require_columns(df, [label_column])
    row_count = df.count()
    fraud_count = df.filter(F.col(label_column) == 1).count()
    return {
        "row_count": int(row_count),
        "column_count": int(len(df.columns)),
        "fraud_count": int(fraud_count),
        "fraud_rate": float(fraud_count / row_count) if row_count else 0.0,
    }
