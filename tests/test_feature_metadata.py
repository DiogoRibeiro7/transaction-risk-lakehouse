from __future__ import annotations

from transaction_risk.features.metadata import (
    feature_registry_to_json,
    feature_registry_to_markdown,
    get_feature_registry,
)
from transaction_risk.features.pipeline import build_feature_table


def test_feature_registry_has_unique_names() -> None:
    registry = get_feature_registry()
    names = [entry.name for entry in registry]

    assert len(names) == len(set(names))


def test_feature_registry_required_fields_are_populated() -> None:
    registry = get_feature_registry()

    for entry in registry:
        assert entry.name
        assert entry.group
        assert entry.dtype
        assert entry.description
        assert entry.source_columns
        assert entry.leakage_risk
        assert entry.owner


def test_feature_registry_covers_main_pipeline_features(spark) -> None:
    df = spark.createDataFrame(
        [
            (0, "TRANSFER", 100.0, "C1", 100.0, 0.0, "C2", 0.0, 100.0, 1, 0, "tx1"),
            (1, "PAYMENT", 25.0, "C1", 100.0, 75.0, "M1", 0.0, 0.0, 0, 0, "tx2"),
            (2, "CASH_OUT", 50.0, "C3", 50.0, 0.0, "C2", 100.0, 150.0, 0, 0, "tx3"),
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

    result = build_feature_table(df)
    registry_names = {entry.name for entry in get_feature_registry()}
    derived_columns = set(result.columns) - set(df.columns)

    assert derived_columns.issubset(registry_names)


def test_feature_registry_exports_are_non_empty() -> None:
    markdown = feature_registry_to_markdown()
    json_output = feature_registry_to_json()

    assert "# Feature Registry" in markdown
    assert "amount_log1p" in markdown
    assert '"name": "amount_log1p"' in json_output
