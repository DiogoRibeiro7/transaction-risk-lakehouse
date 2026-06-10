"""Threshold selection utilities."""

from __future__ import annotations

from collections.abc import Sequence

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from transaction_risk.validation.schema import require_columns


def threshold_for_alert_rate(
    scored_df: DataFrame,
    alert_rate: float,
    probability_column: str = "fraud_probability",
) -> float:
    """Select a probability threshold that approximately produces a target alert rate."""
    require_columns(scored_df, [probability_column])
    if not 0 < alert_rate < 1:
        raise ValueError("alert_rate must be between 0 and 1.")

    quantile = 1.0 - alert_rate
    threshold = scored_df.approxQuantile(probability_column, [quantile], 0.001)[0]
    return float(threshold)


def expected_value_at_threshold(
    scored_df: DataFrame,
    threshold: float,
    amount_column: str = "amount",
    label_column: str = "isFraud",
    score_column: str = "fraud_probability",
    fraud_loss_rate: float = 1.0,
    false_positive_review_cost: float = 1.0,
    true_positive_recovery_rate: float = 1.0,
) -> dict[str, float | int]:
    """Compute business-style expected value metrics at a threshold."""
    require_columns(scored_df, [amount_column, label_column, score_column])
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    alerted = scored_df.withColumn("is_alert", (F.col(score_column) >= threshold).cast("int"))
    aggregated = alerted.agg(
        F.sum(((F.col("is_alert") == 1) & (F.col(label_column) == 1)).cast("int")).alias("tp"),
        F.sum(((F.col("is_alert") == 1) & (F.col(label_column) == 0)).cast("int")).alias("fp"),
        F.sum(((F.col("is_alert") == 0) & (F.col(label_column) == 1)).cast("int")).alias("fn"),
        F.sum(
            F.when((F.col("is_alert") == 1) & (F.col(label_column) == 1), F.col(amount_column)).otherwise(0.0)
        ).alias("tp_amount"),
        F.sum(
            F.when((F.col("is_alert") == 0) & (F.col(label_column) == 1), F.col(amount_column)).otherwise(0.0)
        ).alias("fn_amount"),
    ).collect()[0]

    tp = int(aggregated["tp"] or 0)
    fp = int(aggregated["fp"] or 0)
    fn = int(aggregated["fn"] or 0)
    tp_amount = float(aggregated["tp_amount"] or 0.0)
    fn_amount = float(aggregated["fn_amount"] or 0.0)

    recovered_fraud_value = tp_amount * true_positive_recovery_rate
    missed_fraud_loss = fn_amount * fraud_loss_rate
    review_cost = float(tp + fp) * false_positive_review_cost
    expected_value = recovered_fraud_value - missed_fraud_loss - review_cost

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tp_amount": tp_amount,
        "fn_amount": fn_amount,
        "reviewed_alert_count": int(tp + fp),
        "recovered_fraud_value": float(recovered_fraud_value),
        "missed_fraud_loss": float(missed_fraud_loss),
        "review_cost": float(review_cost),
        "expected_value": float(expected_value),
    }


def optimize_threshold_by_expected_value(
    scored_df: DataFrame,
    candidate_thresholds: Sequence[float],
    amount_column: str = "amount",
    label_column: str = "isFraud",
    score_column: str = "fraud_probability",
    fraud_loss_rate: float = 1.0,
    false_positive_review_cost: float = 1.0,
    true_positive_recovery_rate: float = 1.0,
) -> tuple[float, dict[str, float | int]]:
    """Select the threshold with the highest expected value."""
    if not candidate_thresholds:
        raise ValueError("candidate_thresholds must not be empty.")

    best_threshold: float | None = None
    best_metrics: dict[str, float | int] | None = None

    for threshold in candidate_thresholds:
        metrics = expected_value_at_threshold(
            scored_df=scored_df,
            threshold=float(threshold),
            amount_column=amount_column,
            label_column=label_column,
            score_column=score_column,
            fraud_loss_rate=fraud_loss_rate,
            false_positive_review_cost=false_positive_review_cost,
            true_positive_recovery_rate=true_positive_recovery_rate,
        )
        if best_metrics is None or (
            float(metrics["expected_value"]),
            -float(metrics["threshold"]),
        ) > (
            float(best_metrics["expected_value"]),
            -float(best_metrics["threshold"]),
        ):
            best_threshold = float(threshold)
            best_metrics = metrics

    assert best_threshold is not None
    assert best_metrics is not None
    return best_threshold, best_metrics


def add_alert_flag(
    scored_df: DataFrame,
    threshold: float,
    probability_column: str = "fraud_probability",
    output_column: str = "is_alert",
) -> DataFrame:
    """Add an alert flag based on a fraud probability threshold."""
    require_columns(scored_df, [probability_column])
    return scored_df.withColumn(output_column, (F.col(probability_column) >= threshold).cast("int"))


def add_top_k_alert_flag(
    scored_df: DataFrame,
    k: int,
    probability_column: str = "fraud_probability",
    output_column: str = "is_top_k_alert",
) -> DataFrame:
    """Flag the top-k highest-risk transactions."""
    require_columns(scored_df, [probability_column])
    if k <= 0:
        raise ValueError("k must be positive.")

    window = Window.orderBy(F.col(probability_column).desc())
    return (
        scored_df.withColumn("risk_rank", F.row_number().over(window))
        .withColumn(output_column, (F.col("risk_rank") <= k).cast("int"))
    )
