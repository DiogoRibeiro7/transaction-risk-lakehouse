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
    _, silver = ingest_paysim(
        spark=spark,
        input_path=args.input,
        bronze_path=args.bronze,
        silver_path=args.silver,
        table_format=_table_format(args),
    )

    if args.write_quality_report:
        from transaction_risk.validation.expectations import (
            run_transaction_expectation_suite,
            write_expectation_report,
        )

        suite = run_transaction_expectation_suite(silver, dataset_name="paysim_silver")
        write_expectation_report(suite, args.write_quality_report)
        if args.strict_quality and not suite.success:
            spark.stop()
            raise SystemExit(
                f"Data quality checks failed: {suite.failure_count} expectation(s) did not pass. "
                f"See {args.write_quality_report} for details."
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
    from transaction_risk.models.artifact import save_scoring_artifact
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
    active_validation = scored_validation
    active_test = scored_test
    active_probability_column = "fraud_probability"
    calibrator = None

    calibration_rows: list[dict[str, float | int]] | None = None
    if args.calibrate_probabilities:
        from transaction_risk.models.calibration import (
            apply_calibrator,
            brier_score,
            calibration_table,
            expected_calibration_error,
            fit_isotonic_calibrator,
            fit_platt_calibrator,
        )

        if args.calibration_method == "isotonic":
            calibrator = fit_isotonic_calibrator(scored_validation)
        else:
            calibrator = fit_platt_calibrator(scored_validation)

        active_validation = apply_calibrator(
            scored_validation,
            calibrator,
            output_column="calibrated_fraud_probability",
        )
        active_test = apply_calibrator(
            scored_test,
            calibrator,
            output_column="calibrated_fraud_probability",
        )
        active_probability_column = "calibrated_fraud_probability"

    threshold_metrics: dict[str, float | int] = {}
    if args.threshold_strategy == "expected-value":
        from transaction_risk.models.thresholding import optimize_threshold_by_expected_value

        candidate_thresholds = [threshold / 100.0 for threshold in range(1, 100)]
        threshold, threshold_metrics = optimize_threshold_by_expected_value(
            active_validation,
            candidate_thresholds=candidate_thresholds,
            amount_column=args.amount_column,
            score_column=active_probability_column,
            fraud_loss_rate=args.fraud_loss_rate,
            false_positive_review_cost=args.false_positive_review_cost,
            true_positive_recovery_rate=args.true_positive_recovery_rate,
        )
    else:
        threshold = threshold_for_alert_rate(
            active_validation,
            alert_rate=args.alert_rate,
            probability_column=active_probability_column,
        )

    metrics: dict[str, object] = dict(
        evaluate_scored_model(
            active_test,
            top_k=args.top_k,
            threshold=threshold,
            probability_column=active_probability_column,
        )
    )
    metrics["selected_threshold"] = threshold
    metrics["model_type"] = args.model_type
    metrics["threshold_strategy"] = args.threshold_strategy
    metrics["probability_column"] = active_probability_column
    if threshold_metrics:
        for key, value in threshold_metrics.items():
            metrics[f"validation_{key}"] = value

    if args.calibrate_probabilities:
        from transaction_risk.models.calibration import (
            brier_score,
            calibration_table,
            expected_calibration_error,
        )

        metrics["calibration_method"] = args.calibration_method
        metrics["brier_score_uncalibrated"] = brier_score(scored_test, "fraud_probability")
        metrics["brier_score_calibrated"] = brier_score(active_test, active_probability_column)
        metrics["expected_calibration_error_uncalibrated"] = expected_calibration_error(
            scored_test,
            "fraud_probability",
        )
        metrics["expected_calibration_error_calibrated"] = expected_calibration_error(
            active_test,
            active_probability_column,
        )
        calibration_rows = calibration_table(active_test, active_probability_column)

    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    save_scoring_artifact(model, args.model_output, calibrator=calibrator)

    report: dict[str, object] = dict(metrics)
    if calibration_rows is not None:
        report["calibration_table"] = calibration_rows
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    from transaction_risk.models.registry import register_model

    register_model(
        model_path=args.model_output,
        registry_path=args.registry,
        metrics={
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        },
        feature_table_path=args.input,
        model_type=args.model_type,
        threshold=threshold,
        notes=f"calibration={args.calibration_method}" if calibrator is not None else None,
    )
    spark.stop()


def list_models_command(args: argparse.Namespace) -> None:
    """CLI handler for listing registered models."""
    from transaction_risk.models.registry import load_registry

    entries = load_registry(args.registry)
    if not entries:
        print(f"No models registered in {args.registry}.")
        return

    for entry in entries:
        notes = f" notes={entry.notes}" if entry.notes else ""
        print(
            f"v{entry.version} {entry.model_type} threshold={entry.threshold:.4f} "
            f"registered_at={entry.registered_at} model_path={entry.model_path}{notes}"
        )


def score_batch_command(args: argparse.Namespace) -> None:
    """CLI handler for batch scoring of feature or transaction tables."""
    from transaction_risk.scoring.batch import score_feature_table, score_transaction_table

    spark = create_spark_session_from_yaml(_spark_config_path(args))
    score_function = (
        score_transaction_table if args.input_kind == "transactions" else score_feature_table
    )
    score_function(
        spark=spark,
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        threshold=args.threshold,
        table_format=_table_format(args),
    )
    spark.stop()


def benchmark_models_command(args: argparse.Namespace) -> None:
    """CLI handler for the model benchmark."""
    from transaction_risk.experiments.benchmark import run_model_benchmark

    spark = create_spark_session_from_yaml(_spark_config_path(args))
    model_types = (
        [model_type.strip() for model_type in args.model_types.split(",") if model_type.strip()]
        if args.model_types
        else None
    )
    run_model_benchmark(
        spark=spark,
        feature_path=args.input,
        output_path=args.output,
        model_types=model_types,
        top_k=args.top_k,
        alert_rate=args.alert_rate,
        table_format=_table_format(args),
    )
    spark.stop()


def monitor_command(args: argparse.Namespace) -> None:
    """CLI handler for building monitoring reports from scored tables."""
    from transaction_risk.monitoring.report import (
        build_monitoring_report,
        write_monitoring_report_html,
        write_monitoring_report_json,
        write_monitoring_report_markdown,
    )

    spark = create_spark_session_from_yaml(_spark_config_path(args))
    reference = read_table(spark, args.reference, table_format=_table_format(args))
    current = read_table(spark, args.current, table_format=_table_format(args))

    feature_columns = (
        [column.strip() for column in args.feature_columns.split(",") if column.strip()]
        if args.feature_columns
        else None
    )
    report = build_monitoring_report(
        reference_df=reference,
        current_df=current,
        feature_columns=feature_columns,
        time_bucket_size=args.time_bucket_size,
    )
    if args.output_json:
        write_monitoring_report_json(report, args.output_json)
    if args.output_md:
        write_monitoring_report_markdown(report, args.output_md)
    if args.output_html:
        write_monitoring_report_html(report, args.output_html)
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
    ingest_parser.add_argument(
        "--write-quality-report",
        help="Write a data expectation report (JSON or Markdown by file suffix)",
    )
    ingest_parser.add_argument(
        "--strict-quality",
        action="store_true",
        help="Fail ingestion when expectation checks do not pass",
    )
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
    train_parser.add_argument("--calibrate-probabilities", action="store_true")
    train_parser.add_argument("--calibration-method", default="platt", choices=["platt", "isotonic"])
    train_parser.add_argument("--registry", default="models/registry.jsonl")
    train_parser.set_defaults(func=train_command)

    list_models_parser = subparsers.add_parser("list-models", help="List registered model versions")
    list_models_parser.add_argument("--registry", default="models/registry.jsonl")
    list_models_parser.set_defaults(func=list_models_command)

    score_batch_parser = subparsers.add_parser(
        "score-batch",
        help="Score a materialized feature or transaction table with a saved model",
    )
    score_batch_parser.add_argument("--input", required=True)
    score_batch_parser.add_argument("--model", required=True)
    score_batch_parser.add_argument("--output", required=True)
    score_batch_parser.add_argument("--threshold", type=float, default=0.5)
    score_batch_parser.add_argument("--input-kind", default="features", choices=["features", "transactions"])
    score_batch_parser.add_argument("--table-format", choices=["parquet", "delta"])
    score_batch_parser.set_defaults(func=score_batch_command)

    benchmark_parser = subparsers.add_parser(
        "benchmark-models",
        help="Benchmark model types under one shared temporal split",
    )
    benchmark_parser.add_argument("--input", required=True)
    benchmark_parser.add_argument("--output", required=True)
    benchmark_parser.add_argument("--model-types", help="Comma-separated model types to benchmark")
    benchmark_parser.add_argument("--top-k", type=int, default=100)
    benchmark_parser.add_argument("--alert-rate", type=float, default=0.01)
    benchmark_parser.add_argument("--table-format", choices=["parquet", "delta"])
    benchmark_parser.set_defaults(func=benchmark_models_command)

    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Build drift and performance monitoring reports from scored tables",
    )
    monitor_parser.add_argument("--reference", required=True)
    monitor_parser.add_argument("--current", required=True)
    monitor_parser.add_argument("--output-json")
    monitor_parser.add_argument("--output-md")
    monitor_parser.add_argument("--output-html")
    monitor_parser.add_argument("--feature-columns", help="Comma-separated numeric columns for PSI")
    monitor_parser.add_argument("--time-bucket-size", type=int, default=24)
    monitor_parser.add_argument("--table-format", choices=["parquet", "delta"])
    monitor_parser.set_defaults(func=monitor_command)

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
