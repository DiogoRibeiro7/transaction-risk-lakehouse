from __future__ import annotations

import pytest

from transaction_risk.features.pipeline import build_feature_table
from transaction_risk.models.artifact import save_scoring_artifact
from transaction_risk.models.calibration import fit_platt_calibrator
from transaction_risk.models.evaluation import add_positive_probability
from transaction_risk.models.spark_ml import train_model
from transaction_risk.scoring.batch import (
    score_feature_table,
    score_features_dataframe,
    score_transaction_table,
)
from transaction_risk.spark.io import write_table


def _silver_transactions(spark):
    rows = []
    for index in range(20):
        is_fraud = 1 if index % 5 == 0 else 0
        amount = 50_000.0 + index * 1_000.0 if is_fraud else 100.0 + index
        rows.append(
            (
                index + 1,
                "TRANSFER" if is_fraud else "PAYMENT",
                amount,
                f"C{index:03d}",
                amount,
                0.0,
                f"M{index:03d}",
                0.0,
                0.0,
                is_fraud,
            )
        )
    return spark.createDataFrame(
        rows,
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
        ],
    )


@pytest.fixture(scope="module")
def trained_model_path(spark, tmp_path_factory):
    transactions = _silver_transactions(spark)
    features = build_feature_table(transactions)
    model = train_model(features)
    model_path = tmp_path_factory.mktemp("models") / "fraud_risk_pipeline"
    model.write().overwrite().save(str(model_path))
    return str(model_path)


@pytest.fixture(scope="module")
def calibrated_model_path(spark, tmp_path_factory):
    transactions = _silver_transactions(spark)
    features = build_feature_table(transactions)
    model = train_model(features)
    scored = add_positive_probability(model.transform(features))
    calibrator = fit_platt_calibrator(scored)
    model_path = tmp_path_factory.mktemp("models") / "fraud_risk_calibrated_artifact"
    save_scoring_artifact(model, model_path, calibrator=calibrator)
    return str(model_path)


def test_score_features_dataframe_adds_score_alert_and_threshold(spark, trained_model_path) -> None:
    from pyspark.ml.pipeline import PipelineModel

    transactions = _silver_transactions(spark)
    features = build_feature_table(transactions)
    model = PipelineModel.load(trained_model_path)

    scored = score_features_dataframe(features, model, threshold=0.5)

    assert "fraud_probability" in scored.columns
    assert "is_alert" in scored.columns
    assert "selected_threshold" in scored.columns
    assert "features" not in scored.columns
    assert "rawPrediction" not in scored.columns
    assert scored.count() == features.count()

    row = scored.select("selected_threshold").first()
    assert row["selected_threshold"] == 0.5


def test_score_features_dataframe_rejects_invalid_threshold(spark, trained_model_path) -> None:
    from pyspark.ml.pipeline import PipelineModel

    transactions = _silver_transactions(spark)
    features = build_feature_table(transactions)
    model = PipelineModel.load(trained_model_path)

    with pytest.raises(ValueError):
        score_features_dataframe(features, model, threshold=1.5)


def test_score_feature_table_end_to_end(spark, trained_model_path, repo_tmp_path) -> None:
    transactions = _silver_transactions(spark)
    features = build_feature_table(transactions)
    input_path = repo_tmp_path / "gold_features"
    output_path = repo_tmp_path / "scored_features"
    write_table(features, input_path)

    score_feature_table(
        spark=spark,
        input_path=input_path,
        model_path=trained_model_path,
        output_path=output_path,
        threshold=0.5,
    )

    scored = spark.read.parquet(str(output_path))
    assert scored.count() == features.count()
    assert {"fraud_probability", "is_alert", "selected_threshold"} <= set(scored.columns)


def test_score_transaction_table_builds_features_first(spark, trained_model_path, repo_tmp_path) -> None:
    transactions = _silver_transactions(spark)
    input_path = repo_tmp_path / "silver_transactions"
    output_path = repo_tmp_path / "scored_transactions"
    write_table(transactions, input_path)

    score_transaction_table(
        spark=spark,
        input_path=input_path,
        model_path=trained_model_path,
        output_path=output_path,
        threshold=0.5,
    )

    scored = spark.read.parquet(str(output_path))
    assert scored.count() == transactions.count()
    assert {"fraud_probability", "is_alert", "selected_threshold"} <= set(scored.columns)
    assert "nameOrig" in scored.columns


def test_score_feature_table_supports_calibrated_artifacts(
    spark,
    calibrated_model_path,
    repo_tmp_path,
) -> None:
    transactions = _silver_transactions(spark)
    features = build_feature_table(transactions)
    input_path = repo_tmp_path / "gold_features_calibrated"
    output_path = repo_tmp_path / "scored_features_calibrated"
    write_table(features, input_path)

    score_feature_table(
        spark=spark,
        input_path=input_path,
        model_path=calibrated_model_path,
        output_path=output_path,
        threshold=0.5,
    )

    scored = spark.read.parquet(str(output_path))
    assert scored.count() == features.count()
    assert {"fraud_probability", "is_alert", "selected_threshold"} <= set(scored.columns)
