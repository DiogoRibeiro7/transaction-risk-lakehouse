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
