from __future__ import annotations

from transaction_risk.models.thresholding import (
    add_alert_flag,
    expected_value_at_threshold,
    optimize_threshold_by_expected_value,
    threshold_for_alert_rate,
)


def test_threshold_for_alert_rate_and_alert_flag(spark) -> None:
    df = spark.createDataFrame([(0.1,), (0.2,), (0.8,), (0.9,)], ["fraud_probability"])
    threshold = threshold_for_alert_rate(df, alert_rate=0.5)
    alerted = add_alert_flag(df, threshold=threshold)

    assert 0.0 <= threshold <= 1.0
    assert "is_alert" in alerted.columns
    assert alerted.filter("is_alert = 1").count() >= 1


def test_expected_value_at_threshold_is_deterministic(spark) -> None:
    scored = spark.createDataFrame(
        [
            (0.9, 1, 100.0),
            (0.8, 0, 10.0),
            (0.4, 1, 50.0),
            (0.2, 0, 5.0),
        ],
        ["fraud_probability", "isFraud", "amount"],
    )

    metrics = expected_value_at_threshold(
        scored,
        threshold=0.5,
        false_positive_review_cost=5.0,
        true_positive_recovery_rate=0.5,
    )

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["recovered_fraud_value"] == 50.0
    assert metrics["missed_fraud_loss"] == 50.0
    assert metrics["review_cost"] == 10.0
    assert metrics["expected_value"] == -10.0


def test_optimize_threshold_by_expected_value_selects_best_candidate(spark) -> None:
    scored = spark.createDataFrame(
        [
            (0.95, 1, 100.0),
            (0.80, 0, 5.0),
            (0.60, 1, 40.0),
            (0.30, 0, 5.0),
        ],
        ["fraud_probability", "isFraud", "amount"],
    )

    threshold, metrics = optimize_threshold_by_expected_value(
        scored,
        candidate_thresholds=[0.5, 0.85],
        false_positive_review_cost=5.0,
        true_positive_recovery_rate=1.0,
    )

    assert threshold == 0.5
    assert metrics["expected_value"] == 125.0
