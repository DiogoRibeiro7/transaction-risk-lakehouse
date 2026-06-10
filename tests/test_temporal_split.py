from __future__ import annotations

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
