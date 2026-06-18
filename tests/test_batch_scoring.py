from __future__ import annotations

import pytest

from transaction_risk.features.pipeline import build_feature_table
from transaction_risk.models.artifact import ScoringArtifact
from transaction_risk.scoring.batch import (
    score_feature_table,
    score_features_dataframe,
    score_transaction_table,
)


def _silver_transactions(spark):
    rows = []
    for index in range(12):
        is_fraud = 1 if index in {2, 5, 8, 11} else 0
        amount = 5_000.0 + index * 250.0 if is_fraud else 100.0 + index
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
def scoring_features_df(spark):
    return build_feature_table(_silver_transactions(spark))


class _FakeScoringModel:
    def transform(self, df):
        from pyspark.sql import functions as F

        return (
            df.withColumn("fraud_probability", F.when(F.col("amount") > 1_000.0, F.lit(0.9)).otherwise(F.lit(0.1)))
            .withColumn("rawPrediction", F.lit("raw"))
            .withColumn("features", F.lit("vector"))
        )


def test_score_features_dataframe_adds_score_alert_and_threshold(
    monkeypatch: pytest.MonkeyPatch,
    scoring_features_df,
) -> None:
    monkeypatch.setattr("transaction_risk.scoring.batch.add_positive_probability", lambda df: df)

    scored = score_features_dataframe(scoring_features_df, _FakeScoringModel(), threshold=0.5)

    assert "fraud_probability" in scored.columns
    assert "is_alert" in scored.columns
    assert "selected_threshold" in scored.columns
    assert "features" not in scored.columns
    assert "rawPrediction" not in scored.columns
    assert scored.count() == scoring_features_df.count()

    row = scored.select("selected_threshold").first()
    assert row["selected_threshold"] == 0.5


def test_score_features_dataframe_rejects_invalid_threshold(scoring_features_df) -> None:
    with pytest.raises(ValueError):
        score_features_dataframe(scoring_features_df, _FakeScoringModel(), threshold=1.5)


@pytest.mark.slow
def test_score_feature_table_end_to_end(
    spark,
    monkeypatch: pytest.MonkeyPatch,
    scoring_features_df,
) -> None:
    features = scoring_features_df
    input_path = "gold_features"
    output_path = "scored_features"
    model_path = "fake_model"
    written: dict[str, object] = {}

    monkeypatch.setattr(
        "transaction_risk.scoring.batch.load_scoring_artifact",
        lambda path: ScoringArtifact(model=_FakeScoringModel()),
    )
    monkeypatch.setattr("transaction_risk.scoring.batch.add_positive_probability", lambda df: df)
    monkeypatch.setattr("transaction_risk.scoring.batch.read_table", lambda spark_session, path, table_format="parquet": features)
    monkeypatch.setattr(
        "transaction_risk.scoring.batch.write_table",
        lambda df, path, table_format="parquet", mode="overwrite", partition_columns=None: written.update(
            {"df": df, "path": path, "table_format": table_format}
        ),
    )

    score_feature_table(
        spark=spark,
        input_path=input_path,
        model_path=model_path,
        output_path=output_path,
        threshold=0.5,
    )

    scored = written["df"]
    assert scored.count() == features.count()
    assert {"fraud_probability", "is_alert", "selected_threshold"} <= set(scored.columns)
    assert written["path"] == output_path


@pytest.mark.slow
def test_score_transaction_table_builds_features_first(
    spark,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transactions = _silver_transactions(spark)
    input_path = "silver_transactions"
    output_path = "scored_transactions"
    model_path = "fake_model"
    written: dict[str, object] = {}

    monkeypatch.setattr(
        "transaction_risk.scoring.batch.load_scoring_artifact",
        lambda path: ScoringArtifact(model=_FakeScoringModel()),
    )
    monkeypatch.setattr("transaction_risk.scoring.batch.add_positive_probability", lambda df: df)
    monkeypatch.setattr(
        "transaction_risk.scoring.batch.read_table",
        lambda spark_session, path, table_format="parquet": transactions,
    )
    monkeypatch.setattr(
        "transaction_risk.scoring.batch.write_table",
        lambda df, path, table_format="parquet", mode="overwrite", partition_columns=None: written.update(
            {"df": df, "path": path, "table_format": table_format}
        ),
    )

    score_transaction_table(
        spark=spark,
        input_path=input_path,
        model_path=model_path,
        output_path=output_path,
        threshold=0.5,
    )

    scored = written["df"]
    assert scored.count() == transactions.count()
    assert {"fraud_probability", "is_alert", "selected_threshold"} <= set(scored.columns)
    assert "nameOrig" in scored.columns
    assert written["path"] == output_path


@pytest.mark.slow
def test_score_feature_table_supports_calibrated_artifacts(
    spark,
    monkeypatch: pytest.MonkeyPatch,
    scoring_features_df,
) -> None:
    features = scoring_features_df
    input_path = "gold_features_calibrated"
    output_path = "scored_features_calibrated"
    model_path = "fake_calibrated_model"
    written: dict[str, object] = {}

    monkeypatch.setattr(
        "transaction_risk.scoring.batch.load_scoring_artifact",
        lambda path: ScoringArtifact(model=_FakeScoringModel(), calibrator=object()),
    )
    monkeypatch.setattr("transaction_risk.scoring.batch.add_positive_probability", lambda df: df)
    monkeypatch.setattr("transaction_risk.scoring.batch.read_table", lambda spark_session, path, table_format="parquet": features)
    monkeypatch.setattr(
        "transaction_risk.scoring.batch.write_table",
        lambda df, path, table_format="parquet", mode="overwrite", partition_columns=None: written.update(
            {"df": df, "path": path, "table_format": table_format}
        ),
    )

    def _fake_apply_calibrator(scored_df, calibrator_model, output_column="calibrated_fraud_probability"):
        from pyspark.sql import functions as F

        return scored_df.withColumn(output_column, F.col("fraud_probability") * F.lit(0.95) + F.lit(0.02))

    monkeypatch.setattr("transaction_risk.scoring.batch.apply_calibrator", _fake_apply_calibrator)

    score_feature_table(
        spark=spark,
        input_path=input_path,
        model_path=model_path,
        output_path=output_path,
        threshold=0.5,
    )

    scored = written["df"]
    assert scored.count() == features.count()
    assert {"fraud_probability", "is_alert", "selected_threshold"} <= set(scored.columns)
    assert "uncalibrated_fraud_probability" in scored.columns
