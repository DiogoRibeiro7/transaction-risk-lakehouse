from __future__ import annotations

import pytest

from transaction_risk.models.calibration import (
    apply_calibrator,
    brier_score,
    calibration_table,
    expected_calibration_error,
    fit_isotonic_calibrator,
    fit_platt_calibrator,
)

pytestmark = [pytest.mark.slow, pytest.mark.spark_slow]


def _validation_frame(spark):
    return spark.createDataFrame(
        [
            (0.05, 0),
            (0.10, 0),
            (0.20, 0),
            (0.35, 0),
            (0.45, 1),
            (0.60, 0),
            (0.70, 1),
            (0.80, 1),
            (0.90, 1),
            (0.95, 1),
        ],
        ["fraud_probability", "isFraud"],
    )


def test_platt_calibrator_outputs_valid_monotone_probabilities(spark) -> None:
    validation = _validation_frame(spark)

    calibrator = fit_platt_calibrator(validation)
    calibrated = apply_calibrator(validation, calibrator)

    assert "calibrated_fraud_probability" in calibrated.columns
    assert "calibration_features" not in calibrated.columns
    assert "calibration_probability" not in calibrated.columns

    rows = calibrated.orderBy("fraud_probability").collect()
    values = [row["calibrated_fraud_probability"] for row in rows]
    assert all(0.0 <= value <= 1.0 for value in values)
    assert values == sorted(values)


def test_isotonic_calibrator_outputs_valid_probabilities(spark) -> None:
    validation = _validation_frame(spark)

    calibrator = fit_isotonic_calibrator(validation)
    calibrated = apply_calibrator(validation, calibrator)

    assert "calibrated_fraud_probability" in calibrated.columns
    assert "calibration_prediction" not in calibrated.columns

    rows = calibrated.orderBy("fraud_probability").collect()
    values = [row["calibrated_fraud_probability"] for row in rows]
    assert all(0.0 <= value <= 1.0 for value in values)
    assert values == sorted(values)


def test_calibration_table_bins_and_counts(spark) -> None:
    scored = spark.createDataFrame(
        [
            (0.05, 0),
            (0.15, 0),
            (0.85, 1),
            (0.95, 1),
            (1.00, 1),
        ],
        ["fraud_probability", "isFraud"],
    )

    table = calibration_table(scored, "fraud_probability", n_bins=10)

    bins = {row["bin"]: row for row in table}
    assert set(bins) == {0, 1, 8, 9}
    assert bins[0]["count"] == 1
    assert bins[9]["count"] == 2
    assert bins[9]["positive_count"] == 2
    assert bins[9]["observed_fraud_rate"] == 1.0
    assert bins[0]["bin_lower"] == 0.0
    assert bins[0]["bin_upper"] == 0.1


def test_calibration_table_rejects_invalid_bins(spark) -> None:
    scored = spark.createDataFrame([(0.5, 1)], ["fraud_probability", "isFraud"])

    with pytest.raises(ValueError):
        calibration_table(scored, "fraud_probability", n_bins=0)


def test_brier_score_is_deterministic(spark) -> None:
    scored = spark.createDataFrame(
        [
            (1.0, 1),
            (0.0, 0),
            (0.5, 1),
            (0.5, 0),
        ],
        ["fraud_probability", "isFraud"],
    )

    assert brier_score(scored, "fraud_probability") == pytest.approx(0.125)


def test_expected_calibration_error_is_deterministic(spark) -> None:
    scored = spark.createDataFrame(
        [
            (0.2, 0),
            (0.3, 1),
            (0.8, 1),
            (0.9, 1),
        ],
        ["fraud_probability", "isFraud"],
    )

    ece = expected_calibration_error(scored, "fraud_probability", n_bins=2)

    assert ece == pytest.approx(0.2)
