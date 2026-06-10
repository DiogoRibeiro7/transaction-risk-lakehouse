"""Temporal splitting utilities."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from transaction_risk.validation.schema import require_columns


def temporal_split(
    df: DataFrame,
    time_column: str = "step",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Split a DataFrame into train, validation, and test by ordered time.

    Parameters
    ----------
    df:
        Input DataFrame.
    time_column:
        Ordered time column.
    train_fraction:
        Fraction of the time range assigned to training.
    validation_fraction:
        Fraction assigned to validation after training.

    Returns
    -------
    tuple[DataFrame, DataFrame, DataFrame]
        Train, validation, and test DataFrames.
    """
    require_columns(df, [time_column])
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1.")

    bounds = df.agg(F.min(time_column).alias("min_time"), F.max(time_column).alias("max_time")).collect()[0]
    min_time = bounds["min_time"]
    max_time = bounds["max_time"]
    if min_time is None or max_time is None:
        raise ValueError("Cannot split an empty DataFrame.")

    time_range = max_time - min_time
    train_cutoff = min_time + time_range * train_fraction
    validation_cutoff = min_time + time_range * (train_fraction + validation_fraction)

    train = df.filter(F.col(time_column) <= train_cutoff)
    validation = df.filter((F.col(time_column) > train_cutoff) & (F.col(time_column) <= validation_cutoff))
    test = df.filter(F.col(time_column) > validation_cutoff)
    return train, validation, test
