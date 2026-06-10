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
