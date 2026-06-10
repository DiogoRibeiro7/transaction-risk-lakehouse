from __future__ import annotations

from transaction_risk.models.thresholding import add_alert_flag, threshold_for_alert_rate


def test_threshold_for_alert_rate_and_alert_flag(spark) -> None:
    df = spark.createDataFrame([(0.1,), (0.2,), (0.8,), (0.9,)], ["fraud_probability"])
    threshold = threshold_for_alert_rate(df, alert_rate=0.5)
    alerted = add_alert_flag(df, threshold=threshold)

    assert 0.0 <= threshold <= 1.0
    assert "is_alert" in alerted.columns
    assert alerted.filter("is_alert = 1").count() >= 1
