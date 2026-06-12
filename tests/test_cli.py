from __future__ import annotations

from pathlib import Path

from transaction_risk.cli import build_parser


def test_ingest_parser_accepts_table_format() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "ingest-paysim",
            "--input",
            "data/raw/paysim.csv",
            "--bronze",
            "data/bronze/paysim",
            "--silver",
            "data/silver/transactions",
            "--table-format",
            "delta",
        ]
    )

    assert args.table_format == "delta"


def test_build_features_parser_accepts_table_format() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "build-features",
            "--input",
            "data/silver/transactions",
            "--output",
            "data/gold/features",
            "--table-format",
            "parquet",
        ]
    )

    assert args.table_format == "parquet"


def test_ingest_ieee_cis_parser_accepts_inputs() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "ingest-ieee-cis",
            "--transaction-input",
            "data/raw/train_transaction.csv",
            "--identity-input",
            "data/raw/train_identity.csv",
            "--bronze",
            "data/bronze/ieee_cis",
            "--silver",
            "data/silver/ieee_cis",
            "--table-format",
            "parquet",
        ]
    )

    assert args.transaction_input == "data/raw/train_transaction.csv"
    assert args.identity_input == "data/raw/train_identity.csv"
    assert args.table_format == "parquet"


def test_train_parser_accepts_table_format() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "train",
            "--input",
            "data/gold/features",
            "--model-output",
            "models/fraud_risk_pipeline",
            "--metrics-output",
            str(Path("reports") / "metrics.json"),
            "--table-format",
            "delta",
        ]
    )

    assert args.table_format == "delta"


def test_train_parser_accepts_expected_value_threshold_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "train",
            "--input",
            "data/gold/features",
            "--model-output",
            "models/fraud_risk_pipeline",
            "--metrics-output",
            "reports/metrics.json",
            "--threshold-strategy",
            "expected-value",
            "--fraud-loss-rate",
            "2.0",
            "--false-positive-review-cost",
            "3.0",
            "--true-positive-recovery-rate",
            "0.75",
        ]
    )

    assert args.threshold_strategy == "expected-value"
    assert args.fraud_loss_rate == 2.0
    assert args.false_positive_review_cost == 3.0
    assert args.true_positive_recovery_rate == 0.75


def test_train_parser_accepts_calibration_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "train",
            "--input",
            "data/gold/features",
            "--model-output",
            "models/fraud_risk_pipeline",
            "--metrics-output",
            "reports/metrics.json",
            "--calibrate-probabilities",
            "--calibration-method",
            "isotonic",
        ]
    )

    assert args.calibrate_probabilities is True
    assert args.calibration_method == "isotonic"


def test_train_parser_calibration_is_off_by_default() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "train",
            "--input",
            "data/gold/features",
            "--model-output",
            "models/fraud_risk_pipeline",
            "--metrics-output",
            "reports/metrics.json",
        ]
    )

    assert args.calibrate_probabilities is False
    assert args.calibration_method == "platt"


def test_score_batch_parser_accepts_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "score-batch",
            "--input",
            "data/silver/transactions",
            "--model",
            "models/fraud_risk_pipeline",
            "--output",
            "data/scored/batch",
            "--threshold",
            "0.3",
            "--input-kind",
            "transactions",
            "--table-format",
            "parquet",
        ]
    )

    assert args.input == "data/silver/transactions"
    assert args.model == "models/fraud_risk_pipeline"
    assert args.output == "data/scored/batch"
    assert args.threshold == 0.3
    assert args.input_kind == "transactions"
    assert args.table_format == "parquet"


def test_score_batch_parser_defaults_to_features_input() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "score-batch",
            "--input",
            "data/gold/features",
            "--model",
            "models/fraud_risk_pipeline",
            "--output",
            "data/scored/batch",
        ]
    )

    assert args.input_kind == "features"
    assert args.threshold == 0.5


def test_list_models_parser_uses_default_registry() -> None:
    parser = build_parser()

    args = parser.parse_args(["list-models"])

    assert args.registry == "models/registry.jsonl"


def test_train_parser_accepts_registry_path() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "train",
            "--input",
            "data/gold/features",
            "--model-output",
            "models/fraud_risk_pipeline",
            "--metrics-output",
            "reports/metrics.json",
            "--registry",
            "models/custom_registry.jsonl",
        ]
    )

    assert args.registry == "models/custom_registry.jsonl"


def test_monitor_parser_accepts_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "monitor",
            "--reference",
            "data/scored/reference",
            "--current",
            "data/scored/current",
            "--output-json",
            "reports/monitoring/report.json",
            "--output-md",
            "reports/monitoring/report.md",
            "--feature-columns",
            "amount,fraud_probability",
            "--time-bucket-size",
            "12",
        ]
    )

    assert args.reference == "data/scored/reference"
    assert args.current == "data/scored/current"
    assert args.output_json == "reports/monitoring/report.json"
    assert args.output_md == "reports/monitoring/report.md"
    assert args.feature_columns == "amount,fraud_probability"
    assert args.time_bucket_size == 12


def test_ingest_parser_accepts_quality_report_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "ingest-paysim",
            "--input",
            "data/raw/paysim.csv",
            "--bronze",
            "data/bronze/paysim",
            "--silver",
            "data/silver/transactions",
            "--write-quality-report",
            "reports/quality/paysim_ingestion.json",
            "--strict-quality",
        ]
    )

    assert args.write_quality_report == "reports/quality/paysim_ingestion.json"
    assert args.strict_quality is True


def test_ingest_parser_quality_report_defaults_off() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "ingest-paysim",
            "--input",
            "data/raw/paysim.csv",
            "--bronze",
            "data/bronze/paysim",
            "--silver",
            "data/silver/transactions",
        ]
    )

    assert args.write_quality_report is None
    assert args.strict_quality is False


def test_benchmark_models_parser_accepts_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "benchmark-models",
            "--input",
            "data/gold/features",
            "--output",
            "reports/benchmark",
            "--model-types",
            "logistic_regression,gbt",
            "--top-k",
            "50",
            "--alert-rate",
            "0.02",
        ]
    )

    assert args.input == "data/gold/features"
    assert args.output == "reports/benchmark"
    assert args.model_types == "logistic_regression,gbt"
    assert args.top_k == 50
    assert args.alert_rate == 0.02


def test_export_feature_registry_parser_accepts_output() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "export-feature-registry",
            "--output",
            "reports/feature_registry.json",
        ]
    )

    assert args.output == "reports/feature_registry.json"
