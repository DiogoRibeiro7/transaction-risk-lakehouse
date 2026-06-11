"""Reproducible model benchmark under a shared temporal split."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from transaction_risk.models.evaluation import add_positive_probability, evaluate_scored_model
from transaction_risk.models.spark_ml import ModelTrainingConfig, train_model
from transaction_risk.models.split import temporal_split
from transaction_risk.models.thresholding import threshold_for_alert_rate
from transaction_risk.spark.io import read_table

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_MODELS = ["logistic_regression", "random_forest", "gbt"]

REQUIRED_RESULT_KEYS = {
    "model_type",
    "roc_auc",
    "pr_auc",
    "precision_at_k",
    "recall_at_k",
    "alert_count",
    "selected_threshold",
}

MARKDOWN_COLUMNS = [
    "model_type",
    "roc_auc",
    "pr_auc",
    "precision_at_k",
    "recall_at_k",
    "alert_count",
    "selected_threshold",
]


def validate_benchmark_result(result: dict) -> None:
    """Check that a benchmark result contains the required schema keys."""
    missing = sorted(REQUIRED_RESULT_KEYS - set(result))
    if missing:
        raise ValueError(f"Benchmark result is missing required keys: {', '.join(missing)}.")


def benchmark_single_model(
    train_df: DataFrame,
    validation_df: DataFrame,
    test_df: DataFrame,
    model_type: str,
    top_k: int = 100,
    alert_rate: float = 0.01,
) -> dict:
    """Train one model type and evaluate it on the shared temporal split."""
    logger.info("Benchmarking model type %s", model_type)
    config = ModelTrainingConfig(model_type=model_type)
    model = train_model(train_df, config=config)

    scored_validation = add_positive_probability(model.transform(validation_df))
    threshold = threshold_for_alert_rate(scored_validation, alert_rate=alert_rate)

    scored_test = add_positive_probability(model.transform(test_df))
    metrics = evaluate_scored_model(scored_test, top_k=top_k, threshold=threshold)

    result = {
        "model_type": model_type,
        "selected_threshold": float(threshold),
        "alert_count": int(metrics["tp"]) + int(metrics["fp"]),
        **metrics,
    }
    validate_benchmark_result(result)
    return result


def benchmark_results_to_markdown(results: list[dict]) -> str:
    """Render benchmark results as a GitHub Markdown table."""
    if not results:
        raise ValueError("results must not be empty.")
    for result in results:
        validate_benchmark_result(result)

    lines = [
        "# Model benchmark",
        "",
        "All models share the same temporal train/validation/test split. "
        "Thresholds are selected on validation data by target alert rate.",
        "",
        "| " + " | ".join(MARKDOWN_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in MARKDOWN_COLUMNS) + " |",
    ]
    for result in results:
        cells = []
        for column in MARKDOWN_COLUMNS:
            value = result[column]
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_benchmark_reports(results: list[dict], output_path: str | Path) -> None:
    """Write benchmark results as metrics.json and metrics.md."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "metrics.md").write_text(benchmark_results_to_markdown(results), encoding="utf-8")
    logger.info("Wrote benchmark reports to %s", output_dir)


def run_model_benchmark(
    spark: SparkSession,
    feature_path: str | Path,
    output_path: str | Path,
    model_types: list[str] | None = None,
    top_k: int = 100,
    alert_rate: float = 0.01,
    table_format: str = "parquet",
) -> list[dict]:
    """Benchmark several model types under one shared temporal split."""
    features = read_table(spark, feature_path, table_format=table_format)
    train_df, validation_df, test_df = temporal_split(features)

    results = [
        benchmark_single_model(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            model_type=model_type,
            top_k=top_k,
            alert_rate=alert_rate,
        )
        for model_type in (model_types or DEFAULT_BENCHMARK_MODELS)
    ]
    write_benchmark_reports(results, output_path)
    return results
