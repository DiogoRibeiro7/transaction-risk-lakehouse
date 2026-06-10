"""Command-line interface for transaction-risk-lakehouse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transaction_risk.spark.io import read_table, write_table
from transaction_risk.spark.session import create_spark_session_from_yaml, load_yaml_config


def _spark_config_path(args: argparse.Namespace) -> str:
    value = getattr(args, "spark_config", None)
    return str(value or "conf/spark.local.yaml")


def _table_format(args: argparse.Namespace) -> str:
    value = getattr(args, "table_format", None)
    if value:
        return str(value)
    config = load_yaml_config(_spark_config_path(args))
    return str(config.get("table_format", "parquet"))


def ingest_paysim_command(args: argparse.Namespace) -> None:
    """CLI handler for PaySim ingestion."""
    from transaction_risk.ingestion.paysim import ingest_paysim

    spark = create_spark_session_from_yaml(_spark_config_path(args))
    ingest_paysim(
        spark=spark,
        input_path=args.input,
        bronze_path=args.bronze,
        silver_path=args.silver,
        table_format=_table_format(args),
    )
    spark.stop()


def ingest_ieee_cis_command(args: argparse.Namespace) -> None:
    """CLI handler for IEEE-CIS ingestion."""
    from transaction_risk.ingestion.ieee_cis import ingest_ieee_cis

    spark = create_spark_session_from_yaml(_spark_config_path(args))
    ingest_ieee_cis(
        spark=spark,
        transaction_path=args.transaction_input,
        identity_path=args.identity_input,
        bronze_path=args.bronze,
        silver_path=args.silver,
        table_format=_table_format(args),
    )
    spark.stop()


def build_features_command(args: argparse.Namespace) -> None:
    """CLI handler for feature table creation."""
    from transaction_risk.features.pipeline import build_feature_table

    spark = create_spark_session_from_yaml(_spark_config_path(args))
    transactions = read_table(spark, args.input, table_format=_table_format(args))
    features = build_feature_table(transactions)
    write_table(features, args.output, table_format=_table_format(args))
    spark.stop()


def export_feature_registry_command(args: argparse.Namespace) -> None:
    """CLI handler for exporting the feature registry."""
    from transaction_risk.features.metadata import (
        feature_registry_to_json,
        feature_registry_to_markdown,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix == ".json":
        output_text = feature_registry_to_json()
    else:
        output_text = feature_registry_to_markdown()
    output_path.write_text(output_text, encoding="utf-8")


def train_command(args: argparse.Namespace) -> None:
    """CLI handler for model training."""
    from transaction_risk.models.evaluation import add_positive_probability, evaluate_scored_model
    from transaction_risk.models.spark_ml import ModelTrainingConfig, train_model
    from transaction_risk.models.split import temporal_split
    from transaction_risk.models.thresholding import threshold_for_alert_rate

    spark = create_spark_session_from_yaml(_spark_config_path(args))
    features = read_table(spark, args.input, table_format=_table_format(args))
    train_df, validation_df, test_df = temporal_split(features)

    config = ModelTrainingConfig(model_type=args.model_type)
    model = train_model(train_df, config=config)
    scored_validation = add_positive_probability(model.transform(validation_df))
    scored_test = add_positive_probability(model.transform(test_df))

    threshold_metrics: dict[str, float | int] = {}
    if args.threshold_strategy == "expected-value":
        from transaction_risk.models.thresholding import optimize_threshold_by_expected_value

        candidate_thresholds = [threshold / 100.0 for threshold in range(1, 100)]
        threshold, threshold_metrics = optimize_threshold_by_expected_value(
            scored_validation,
            candidate_thresholds=candidate_thresholds,
            amount_column=args.amount_column,
            fraud_loss_rate=args.fraud_loss_rate,
            false_positive_review_cost=args.false_positive_review_cost,
            true_positive_recovery_rate=args.true_positive_recovery_rate,
        )
    else:
        threshold = threshold_for_alert_rate(scored_validation, alert_rate=args.alert_rate)

    metrics = evaluate_scored_model(scored_test, top_k=args.top_k, threshold=threshold)
    metrics["selected_threshold"] = threshold
    metrics["model_type"] = args.model_type
    metrics["threshold_strategy"] = args.threshold_strategy
    if threshold_metrics:
        for key, value in threshold_metrics.items():
            metrics[f"validation_{key}"] = value

    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(args.model_output)

    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    spark.stop()


def score_stream_command(args: argparse.Namespace) -> None:
    """CLI handler for local streaming scoring."""
    from transaction_risk.streaming.scoring_job import run_file_stream_scoring

    spark = create_spark_session_from_yaml(_spark_config_path(args))
    run_file_stream_scoring(
        spark=spark,
        input_stream_path=args.input_stream,
        model_path=args.model,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the project CLI parser."""
    parser = argparse.ArgumentParser(description="Transaction risk lakehouse CLI")
    parser.add_argument("--spark-config", default="conf/spark.local.yaml", help="Path to Spark YAML config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest-paysim", help="Ingest PaySim CSV into bronze and silver tables")
    ingest_parser.add_argument("--input", required=True)
    ingest_parser.add_argument("--bronze", required=True)
    ingest_parser.add_argument("--silver", required=True)
    ingest_parser.add_argument("--table-format", choices=["parquet", "delta"])
    ingest_parser.set_defaults(func=ingest_paysim_command)

    ieee_ingest_parser = subparsers.add_parser(
        "ingest-ieee-cis",
        help="Ingest IEEE-CIS transaction and identity CSV files into bronze and silver tables",
    )
    ieee_ingest_parser.add_argument("--transaction-input", required=True)
    ieee_ingest_parser.add_argument("--identity-input", required=True)
    ieee_ingest_parser.add_argument("--bronze", required=True)
    ieee_ingest_parser.add_argument("--silver", required=True)
    ieee_ingest_parser.add_argument("--table-format", choices=["parquet", "delta"])
    ieee_ingest_parser.set_defaults(func=ingest_ieee_cis_command)

    features_parser = subparsers.add_parser("build-features", help="Build model-ready feature table")
    features_parser.add_argument("--input", required=True)
    features_parser.add_argument("--output", required=True)
    features_parser.add_argument("--table-format", choices=["parquet", "delta"])
    features_parser.set_defaults(func=build_features_command)

    registry_parser = subparsers.add_parser(
        "export-feature-registry",
        help="Export feature metadata as Markdown or JSON",
    )
    registry_parser.add_argument("--output", default="reports/feature_registry.md")
    registry_parser.set_defaults(func=export_feature_registry_command)

    train_parser = subparsers.add_parser("train", help="Train and evaluate a Spark ML fraud model")
    train_parser.add_argument("--input", required=True)
    train_parser.add_argument("--model-output", required=True)
    train_parser.add_argument("--metrics-output", required=True)
    train_parser.add_argument("--model-type", default="logistic_regression", choices=["logistic_regression", "random_forest", "gbt"])
    train_parser.add_argument("--top-k", type=int, default=100)
    train_parser.add_argument("--alert-rate", type=float, default=0.01)
    train_parser.add_argument("--table-format", choices=["parquet", "delta"])
    train_parser.add_argument("--threshold-strategy", default="alert-rate", choices=["alert-rate", "expected-value"])
    train_parser.add_argument("--amount-column", default="amount")
    train_parser.add_argument("--fraud-loss-rate", type=float, default=1.0)
    train_parser.add_argument("--false-positive-review-cost", type=float, default=1.0)
    train_parser.add_argument("--true-positive-recovery-rate", type=float, default=1.0)
    train_parser.set_defaults(func=train_command)

    stream_parser = subparsers.add_parser("score-stream", help="Score incoming local CSV files with Structured Streaming")
    stream_parser.add_argument("--input-stream", required=True)
    stream_parser.add_argument("--model", required=True)
    stream_parser.add_argument("--output", required=True)
    stream_parser.add_argument("--checkpoint", required=True)
    stream_parser.add_argument("--threshold", type=float, default=0.5)
    stream_parser.set_defaults(func=score_stream_command)

    return parser


def main() -> None:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
