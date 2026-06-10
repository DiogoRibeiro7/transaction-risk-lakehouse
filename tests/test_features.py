from __future__ import annotations

from transaction_risk.features.pipeline import build_feature_table


def test_build_feature_table_adds_expected_columns(spark) -> None:
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
    expected = {
        "amount_log1p",
        "origin_balance_delta",
        "steps_since_previous_origin_tx",
        "origin_total_tx_count",
        "origin_out_degree",
        "destination_in_degree",
        "edge_frequency",
    }
    assert expected.issubset(set(result.columns))
    assert result.count() == 3
