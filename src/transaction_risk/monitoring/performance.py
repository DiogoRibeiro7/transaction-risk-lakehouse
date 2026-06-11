"""Operational monitoring reports for fraud alerts."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from transaction_risk.validation.schema import require_columns


def _time_bucket(time_column: str, bucket_size: int) -> Column:
    """Build a deterministic integer time-bucket expression."""
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive.")
    return F.floor(F.col(time_column) / bucket_size).cast("int")


def alert_summary(
    scored_df: DataFrame,
    alert_column: str = "is_alert",
    label_column: str = "isFraud",
    amount_column: str = "amount",
) -> dict[str, float | int]:
    """Summarize alert volume, fraud capture, and value capture."""
    require_columns(scored_df, [alert_column, label_column, amount_column])

    row = scored_df.agg(
        F.count(F.lit(1)).alias("transactions"),
        F.sum(F.col(alert_column)).alias("alerts"),
        F.sum(F.col(label_column)).alias("frauds"),
        F.sum((F.col(alert_column) * F.col(label_column)).cast("int")).alias("captured_frauds"),
        F.sum(F.when(F.col(alert_column) == 1, F.col(amount_column)).otherwise(F.lit(0.0))).alias("alert_amount"),
        F.sum(F.when(F.col(label_column) == 1, F.col(amount_column)).otherwise(F.lit(0.0))).alias("fraud_amount"),
        F.sum(
            F.when((F.col(alert_column) == 1) & (F.col(label_column) == 1), F.col(amount_column)).otherwise(F.lit(0.0))
        ).alias("captured_fraud_amount"),
    ).collect()[0]

    transactions = int(row["transactions"] or 0)
    alerts = int(row["alerts"] or 0)
    frauds = int(row["frauds"] or 0)
    captured_frauds = int(row["captured_frauds"] or 0)
    fraud_amount = float(row["fraud_amount"] or 0.0)
    captured_fraud_amount = float(row["captured_fraud_amount"] or 0.0)

    return {
        "transactions": transactions,
        "alerts": alerts,
        "alert_rate": alerts / transactions if transactions else 0.0,
        "frauds": frauds,
        "captured_frauds": captured_frauds,
        "fraud_recall": captured_frauds / frauds if frauds else 0.0,
        "alert_amount": float(row["alert_amount"] or 0.0),
        "fraud_amount": fraud_amount,
        "captured_fraud_amount": captured_fraud_amount,
        "fraud_value_capture_rate": captured_fraud_amount / fraud_amount if fraud_amount else 0.0,
    }


def alert_volume_by_time_bucket(
    scored_df: DataFrame,
    time_column: str = "step",
    bucket_size: int = 24,
    alert_column: str = "is_alert",
) -> list[dict[str, float | int]]:
    """Compute transaction and alert volume per time bucket."""
    require_columns(scored_df, [time_column, alert_column])

    rows = (
        scored_df.withColumn("time_bucket", _time_bucket(time_column, bucket_size))
        .groupBy("time_bucket")
        .agg(
            F.count(F.lit(1)).alias("transactions"),
            F.sum(F.col(alert_column).cast("int")).alias("alerts"),
        )
        .orderBy("time_bucket")
        .collect()
    )

    return [
        {
            "time_bucket": int(row["time_bucket"]),
            "transactions": int(row["transactions"]),
            "alerts": int(row["alerts"] or 0),
            "alert_rate": float((row["alerts"] or 0) / row["transactions"]),
        }
        for row in rows
    ]


def precision_recall_by_time_bucket(
    scored_df: DataFrame,
    time_column: str = "step",
    bucket_size: int = 24,
    alert_column: str = "is_alert",
    label_column: str = "isFraud",
) -> list[dict[str, float | int]]:
    """Compute labeled precision and recall per time bucket."""
    require_columns(scored_df, [time_column, alert_column, label_column])

    rows = (
        scored_df.withColumn("time_bucket", _time_bucket(time_column, bucket_size))
        .groupBy("time_bucket")
        .agg(
            F.sum(F.col(alert_column).cast("int")).alias("alerts"),
            F.sum(F.col(label_column).cast("int")).alias("frauds"),
            F.sum(
                ((F.col(alert_column) == 1) & (F.col(label_column) == 1)).cast("int")
            ).alias("captured_frauds"),
        )
        .orderBy("time_bucket")
        .collect()
    )

    results: list[dict[str, float | int]] = []
    for row in rows:
        alerts = int(row["alerts"] or 0)
        frauds = int(row["frauds"] or 0)
        captured = int(row["captured_frauds"] or 0)
        results.append(
            {
                "time_bucket": int(row["time_bucket"]),
                "alerts": alerts,
                "frauds": frauds,
                "captured_frauds": captured,
                "precision": captured / alerts if alerts else 0.0,
                "recall": captured / frauds if frauds else 0.0,
            }
        )
    return results


def fraud_value_by_time_bucket(
    scored_df: DataFrame,
    time_column: str = "step",
    bucket_size: int = 24,
    alert_column: str = "is_alert",
    label_column: str = "isFraud",
    amount_column: str = "amount",
) -> list[dict[str, float | int]]:
    """Compute fraud value and captured fraud value per time bucket."""
    require_columns(scored_df, [time_column, alert_column, label_column, amount_column])

    rows = (
        scored_df.withColumn("time_bucket", _time_bucket(time_column, bucket_size))
        .groupBy("time_bucket")
        .agg(
            F.sum(
                F.when(F.col(label_column) == 1, F.col(amount_column)).otherwise(F.lit(0.0))
            ).alias("fraud_amount"),
            F.sum(
                F.when(
                    (F.col(alert_column) == 1) & (F.col(label_column) == 1),
                    F.col(amount_column),
                ).otherwise(F.lit(0.0))
            ).alias("captured_fraud_amount"),
        )
        .orderBy("time_bucket")
        .collect()
    )

    results: list[dict[str, float | int]] = []
    for row in rows:
        fraud_amount = float(row["fraud_amount"] or 0.0)
        captured_amount = float(row["captured_fraud_amount"] or 0.0)
        results.append(
            {
                "time_bucket": int(row["time_bucket"]),
                "fraud_amount": fraud_amount,
                "captured_fraud_amount": captured_amount,
                "fraud_value_capture_rate": captured_amount / fraud_amount if fraud_amount else 0.0,
            }
        )
    return results
