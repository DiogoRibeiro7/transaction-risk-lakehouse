from __future__ import annotations

import pytest

from transaction_risk.models.split import temporal_split


def test_temporal_split_preserves_time_order(spark) -> None:
    df = spark.createDataFrame([(i, float(i), i % 2) for i in range(100)], ["step", "amount", "isFraud"])
    train, validation, test = temporal_split(df, time_column="step", train_fraction=0.7, validation_fraction=0.15)

    max_train = train.agg({"step": "max"}).collect()[0][0]
    min_validation = validation.agg({"step": "min"}).collect()[0][0]
    max_validation = validation.agg({"step": "max"}).collect()[0][0]
    min_test = test.agg({"step": "min"}).collect()[0][0]

    assert max_train < min_validation
    assert max_validation < min_test
    assert train.count() + validation.count() + test.count() == 100


def test_temporal_split_handles_sparse_time_buckets_without_empty_windows(spark) -> None:
    df = spark.createDataFrame(
        [
            (0, 10.0, 0),
            (0, 11.0, 0),
            (100, 20.0, 1),
            (100, 21.0, 0),
            (200, 30.0, 1),
        ],
        ["step", "amount", "isFraud"],
    )

    train, validation, test = temporal_split(df, time_column="step", train_fraction=0.5, validation_fraction=0.25)

    assert train.count() == 2
    assert validation.count() == 2
    assert test.count() == 1


def test_temporal_split_rejects_insufficient_distinct_times(spark) -> None:
    df = spark.createDataFrame(
        [(0, 10.0, 0), (0, 12.0, 1), (100, 20.0, 0)],
        ["step", "amount", "isFraud"],
    )

    with pytest.raises(ValueError, match="at least three distinct time values"):
        temporal_split(df, time_column="step")
