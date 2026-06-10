"""Schema validation helpers."""

from __future__ import annotations

from pyspark.sql import DataFrame


def require_columns(df: DataFrame, required_columns: list[str]) -> None:
    """Raise an error when a DataFrame is missing required columns."""
    if not required_columns:
        raise ValueError("required_columns must not be empty.")

    existing = set(df.columns)
    missing = [column for column in required_columns if column not in existing]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def require_binary_label(df: DataFrame, label_column: str) -> None:
    """Validate that a label column exists and contains only 0/1 values."""
    require_columns(df, [label_column])
    invalid_count = df.filter(~df[label_column].isin(0, 1)).count()
    if invalid_count > 0:
        raise ValueError(f"Column `{label_column}` contains values outside {{0, 1}}.")
