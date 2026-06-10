"""Model evaluation utilities for fraud detection."""

from __future__ import annotations

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from transaction_risk.validation.schema import require_columns


def add_positive_probability(
    df: DataFrame,
    probability_column: str = "probability",
    output_column: str = "fraud_probability",
) -> DataFrame:
    """Extract the probability of the positive class from Spark probability vectors."""
    require_columns(df, [probability_column])
    return df.withColumn(output_column, vector_to_array(F.col(probability_column))[1])


def binary_auc_metrics(
    scored_df: DataFrame,
    label_column: str = "isFraud",
    probability_column: str = "fraud_probability",
) -> dict[str, float]:
    """Compute ROC-AUC and PR-AUC."""
    require_columns(scored_df, [label_column, probability_column])

    roc_evaluator = BinaryClassificationEvaluator(
        labelCol=label_column,
        rawPredictionCol=probability_column,
        metricName="areaUnderROC",
    )
    pr_evaluator = BinaryClassificationEvaluator(
        labelCol=label_column,
        rawPredictionCol=probability_column,
        metricName="areaUnderPR",
    )
    return {
        "roc_auc": float(roc_evaluator.evaluate(scored_df)),
        "pr_auc": float(pr_evaluator.evaluate(scored_df)),
    }


def precision_recall_at_k(
    scored_df: DataFrame,
    k: int,
    label_column: str = "isFraud",
    probability_column: str = "fraud_probability",
) -> dict[str, float | int]:
    """Compute precision and recall among the top-k highest-risk transactions."""
    require_columns(scored_df, [label_column, probability_column])
    if k <= 0:
        raise ValueError("k must be positive.")

    ranking_window = Window.orderBy(F.col(probability_column).desc())
    ranked = scored_df.withColumn("risk_rank", F.row_number().over(ranking_window))
    top_k = ranked.filter(F.col("risk_rank") <= k)

    true_positives = top_k.filter(F.col(label_column) == 1).count()
    total_positives = scored_df.filter(F.col(label_column) == 1).count()

    return {
        "k": int(k),
        "precision_at_k": float(true_positives / k),
        "recall_at_k": float(true_positives / total_positives) if total_positives else 0.0,
        "true_positives_at_k": int(true_positives),
        "total_positives": int(total_positives),
    }


def confusion_matrix_at_threshold(
    scored_df: DataFrame,
    threshold: float,
    label_column: str = "isFraud",
    probability_column: str = "fraud_probability",
) -> dict[str, int | float]:
    """Compute confusion-matrix counts at a probability threshold."""
    require_columns(scored_df, [label_column, probability_column])
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    predicted = scored_df.withColumn("alert", (F.col(probability_column) >= threshold).cast("int"))
    aggregated = predicted.agg(
        F.sum(((F.col("alert") == 1) & (F.col(label_column) == 1)).cast("int")).alias("tp"),
        F.sum(((F.col("alert") == 1) & (F.col(label_column) == 0)).cast("int")).alias("fp"),
        F.sum(((F.col("alert") == 0) & (F.col(label_column) == 0)).cast("int")).alias("tn"),
        F.sum(((F.col("alert") == 0) & (F.col(label_column) == 1)).cast("int")).alias("fn"),
    ).collect()[0]

    tp = int(aggregated["tp"] or 0)
    fp = int(aggregated["fp"] or 0)
    tn = int(aggregated["tn"] or 0)
    fn = int(aggregated["fn"] or 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
    }


def evaluate_scored_model(
    scored_df: DataFrame,
    top_k: int = 100,
    threshold: float = 0.5,
    label_column: str = "isFraud",
    probability_column: str = "fraud_probability",
) -> dict[str, float | int]:
    """Compute the standard fraud-model evaluation report."""
    metrics: dict[str, float | int] = {}
    metrics.update(binary_auc_metrics(scored_df, label_column, probability_column))
    metrics.update(precision_recall_at_k(scored_df, top_k, label_column, probability_column))
    metrics.update(confusion_matrix_at_threshold(scored_df, threshold, label_column, probability_column))
    return metrics
