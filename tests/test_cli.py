from __future__ import annotations

import json
from pathlib import Path

from transaction_risk.cli import build_parser, train_command


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
            "--output-html",
            "reports/monitoring/report.html",
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
    assert args.output_html == "reports/monitoring/report.html"
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


def test_train_command_writes_deployable_calibrated_artifact(
    spark,
    monkeypatch,
) -> None:
    features = spark.createDataFrame(
        [
            (1, "PAYMENT", 11.0, "C1", 100.0, 89.0, "D1", 0.0, 11.0, 0, 0, "tx1"),
            (2, "PAYMENT", 12.0, "C2", 100.0, 88.0, "D2", 0.0, 12.0, 0, 0, "tx2"),
            (3, "PAYMENT", 13.0, "C3", 100.0, 87.0, "D3", 0.0, 13.0, 0, 0, "tx3"),
            (4, "TRANSFER", 280.0, "C4", 280.0, 0.0, "D4", 0.0, 280.0, 1, 0, "tx4"),
            (5, "PAYMENT", 15.0, "C1", 89.0, 74.0, "D1", 11.0, 26.0, 0, 0, "tx5"),
            (6, "PAYMENT", 16.0, "C2", 88.0, 72.0, "D2", 12.0, 28.0, 0, 0, "tx6"),
            (7, "PAYMENT", 17.0, "C3", 87.0, 70.0, "D3", 13.0, 30.0, 0, 0, "tx7"),
            (8, "TRANSFER", 320.0, "C5", 320.0, 0.0, "D4", 280.0, 600.0, 1, 0, "tx8"),
            (9, "PAYMENT", 19.0, "C1", 74.0, 55.0, "D1", 26.0, 45.0, 0, 0, "tx9"),
            (10, "PAYMENT", 20.0, "C2", 72.0, 52.0, "D2", 28.0, 48.0, 0, 0, "tx10"),
            (11, "PAYMENT", 21.0, "C3", 70.0, 49.0, "D3", 30.0, 51.0, 0, 0, "tx11"),
            (12, "TRANSFER", 360.0, "C6", 360.0, 0.0, "D4", 600.0, 960.0, 1, 0, "tx12"),
            (13, "PAYMENT", 23.0, "C1", 55.0, 32.0, "D1", 45.0, 68.0, 0, 0, "tx13"),
            (14, "PAYMENT", 24.0, "C2", 52.0, 28.0, "D2", 48.0, 72.0, 0, 0, "tx14"),
            (15, "TRANSFER", 410.0, "C7", 410.0, 0.0, "D5", 0.0, 410.0, 1, 0, "tx15"),
            (16, "PAYMENT", 26.0, "C3", 49.0, 23.0, "D3", 51.0, 77.0, 0, 0, "tx16"),
            (17, "TRANSFER", 440.0, "C8", 440.0, 0.0, "D5", 410.0, 850.0, 1, 0, "tx17"),
            (18, "PAYMENT", 28.0, "C1", 32.0, 4.0, "D1", 68.0, 96.0, 0, 0, "tx18"),
            (19, "PAYMENT", 29.0, "C2", 28.0, 0.0, "D2", 72.0, 101.0, 0, 0, "tx19"),
            (20, "TRANSFER", 480.0, "C9", 480.0, 0.0, "D5", 850.0, 1330.0, 1, 0, "tx20"),
        ],
        [
            "step",
            "type",
            "amount",
            "nameOrig",
            "oldbalanceOrg",
            "newbalanceOrig",
            "nameDest",
            "oldbalanceDest",
            "newbalanceDest",
            "isFraud",
            "isFlaggedFraud",
            "transaction_id",
        ],
    )
    feature_path = Path("data/gold/features")
    model_path = Path("models/artifact_model")
    metrics_path = Path(".tmp") / "train_metrics.json"
    registry_path = Path("models/registry.jsonl")

    monkeypatch.setattr("transaction_risk.cli.create_spark_session_from_yaml", lambda path: spark)
    monkeypatch.setattr(spark, "stop", lambda: None)
    monkeypatch.setattr("transaction_risk.cli.read_table", lambda spark_session, path, table_format: features)
    saved_artifact: dict[str, object] = {}
    registered_model: dict[str, object] = {}

    class FakeModel:
        def transform(self, df):
            from pyspark.sql import functions as F

            return df.withColumn(
                "fraud_probability",
                F.when(F.col("amount") >= 300.0, F.lit(0.9)).otherwise(F.lit(0.1)),
            )

    def fake_save_scoring_artifact(model, output_path, calibrator=None) -> None:
        saved_artifact["output_path"] = output_path
        saved_artifact["has_calibrator"] = calibrator is not None

    def fake_register_model(**kwargs):
        registered_model.update(kwargs)

    def fake_add_positive_probability(df, probability_column="probability", output_column="fraud_probability"):
        return df

    def fake_train_model(train_df, config=None, numeric_features=None):
        return FakeModel()

    def fake_fit_platt_calibrator(validation_scored_df):
        return object()

    def fake_apply_calibrator(scored_df, calibrator_model, output_column="calibrated_fraud_probability"):
        from pyspark.sql import functions as F

        return scored_df.withColumn(output_column, F.col("fraud_probability") * F.lit(0.95) + F.lit(0.02))

    def fake_threshold_for_alert_rate(scored_df, alert_rate, probability_column="fraud_probability"):
        return 0.5

    def fake_evaluate_scored_model(
        scored_df,
        top_k=100,
        threshold=0.5,
        label_column="isFraud",
        probability_column="fraud_probability",
    ):
        assert probability_column == "calibrated_fraud_probability"
        return {
            "roc_auc": 0.91,
            "pr_auc": 0.42,
            "k": top_k,
            "precision_at_k": 0.5,
            "recall_at_k": 0.5,
            "true_positives_at_k": 1,
            "total_positives": 1,
            "threshold": threshold,
            "tp": 1,
            "fp": 0,
            "tn": 1,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
        }

    monkeypatch.setattr("transaction_risk.models.artifact.save_scoring_artifact", fake_save_scoring_artifact)
    monkeypatch.setattr("transaction_risk.models.registry.register_model", fake_register_model)
    monkeypatch.setattr("transaction_risk.models.evaluation.add_positive_probability", fake_add_positive_probability)
    monkeypatch.setattr("transaction_risk.models.evaluation.evaluate_scored_model", fake_evaluate_scored_model)
    monkeypatch.setattr("transaction_risk.models.spark_ml.train_model", fake_train_model)
    monkeypatch.setattr("transaction_risk.models.thresholding.threshold_for_alert_rate", fake_threshold_for_alert_rate)
    monkeypatch.setattr("transaction_risk.models.calibration.fit_platt_calibrator", fake_fit_platt_calibrator)
    monkeypatch.setattr("transaction_risk.models.calibration.apply_calibrator", fake_apply_calibrator)
    monkeypatch.setattr("transaction_risk.models.calibration.brier_score", lambda *args, **kwargs: 0.12)
    monkeypatch.setattr("transaction_risk.models.calibration.expected_calibration_error", lambda *args, **kwargs: 0.03)
    monkeypatch.setattr(
        "transaction_risk.models.calibration.calibration_table",
        lambda *args, **kwargs: [{"bin": 0, "count": 1, "positive_count": 0}],
    )

    args = build_parser().parse_args(
        [
            "train",
            "--input", str(feature_path),
            "--model-output", str(model_path),
            "--metrics-output", str(metrics_path),
            "--table-format",
            "parquet",
            "--top-k",
            "2",
            "--alert-rate",
            "0.4",
            "--calibrate-probabilities",
            "--calibration-method",
            "platt",
            "--registry",
            str(registry_path),
        ]
    )
    train_command(args)

    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert Path(saved_artifact["output_path"]) == model_path
    assert saved_artifact["has_calibrator"] is True
    assert report["calibration_method"] == "platt"
    assert report["probability_column"] == "calibrated_fraud_probability"
    assert "brier_score_calibrated" in report
    assert report["selected_threshold"] == 0.5

    assert Path(registered_model["model_path"]) == model_path
    assert Path(registered_model["registry_path"]) == registry_path
    assert registered_model["notes"] == "calibration=platt"
