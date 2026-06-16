"""Helpers for deterministic historical feature windows."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.window import WindowSpec


def resolve_time_column(df: DataFrame, candidates: list[str] | None = None) -> str:
    """Return the first supported time column present in the DataFrame."""
    for candidate in candidates or ["step", "TransactionDT"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "A supported time column is required for historical features. "
        "Expected one of: step, TransactionDT."
    )


def order_columns(df: DataFrame, time_column: str) -> list[Column]:
    """Build a deterministic ordering for temporal windows."""
    columns: list[Column] = [F.col(time_column)]
    for candidate in ("transaction_id", "TransactionID"):
        if candidate in df.columns:
            columns.append(F.col(candidate))
            break
    return columns


def ordered_window(df: DataFrame, partition_columns: list[str], time_column: str) -> WindowSpec:
    """Build a partitioned temporal window ordered by time and a stable tie-breaker."""
    return Window.partitionBy(*partition_columns).orderBy(*order_columns(df, time_column))
