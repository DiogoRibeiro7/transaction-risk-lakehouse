"""Temporal splitting utilities."""

from __future__ import annotations

import math

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from transaction_risk.validation.schema import require_columns


def temporal_split(
    df: DataFrame,
    time_column: str = "step",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Split a DataFrame into train, validation, and test by ordered time buckets."""
    require_columns(df, [time_column])
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1.")

    distinct_times = [
        row[time_column]
        for row in df.select(time_column).distinct().orderBy(time_column).collect()
    ]
    if not distinct_times:
        raise ValueError("Cannot split an empty DataFrame.")
    if len(distinct_times) < 3:
        raise ValueError(
            "Temporal split requires at least three distinct time values "
            "to create non-empty train, validation, and test windows."
        )

    train_time_count = max(1, min(len(distinct_times) - 2, math.floor(len(distinct_times) * train_fraction)))
    validation_time_count = max(
        1,
        min(
            len(distinct_times) - train_time_count - 1,
            math.floor(len(distinct_times) * validation_fraction),
        ),
    )
    if train_time_count + validation_time_count >= len(distinct_times):
        raise ValueError("Temporal split fractions do not leave any time values for the test window.")

    train_cutoff = distinct_times[train_time_count - 1]
    validation_cutoff = distinct_times[train_time_count + validation_time_count - 1]

    train = df.filter(F.col(time_column) <= F.lit(train_cutoff))
    validation = df.filter(
        (F.col(time_column) > F.lit(train_cutoff)) & (F.col(time_column) <= F.lit(validation_cutoff))
    )
    test = df.filter(F.col(time_column) > F.lit(validation_cutoff))

    if train.limit(1).count() == 0 or validation.limit(1).count() == 0 or test.limit(1).count() == 0:
        raise ValueError("Temporal split produced an empty train, validation, or test window.")
    return train, validation, test
