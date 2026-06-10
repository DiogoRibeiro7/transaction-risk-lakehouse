"""Operational monitoring reports for fraud alerts."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from transaction_risk.validation.schema import require_columns


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
