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
