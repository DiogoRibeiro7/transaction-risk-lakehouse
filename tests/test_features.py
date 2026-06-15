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
    assert len(result.select("transaction_id").collect()) == 3


def test_build_feature_table_uses_only_historical_values(spark) -> None:
    df = spark.createDataFrame(
        [
            (1, "PAYMENT", 10.0, "C1", 100.0, 90.0, "D1", 0.0, 10.0, 0, 0, "tx1"),
            (2, "TRANSFER", 20.0, "C1", 90.0, 70.0, "D2", 0.0, 20.0, 1, 0, "tx2"),
            (3, "CASH_OUT", 30.0, "C1", 70.0, 40.0, "D1", 10.0, 40.0, 0, 0, "tx3"),
            (4, "TRANSFER", 40.0, "C2", 200.0, 160.0, "D1", 40.0, 80.0, 1, 0, "tx4"),
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
    rows = {row["transaction_id"]: row.asDict() for row in result.collect()}

    assert rows["tx1"]["origin_total_tx_count"] == 0
    assert rows["tx1"]["origin_out_degree"] == 0
    assert rows["tx1"]["destination_historical_fraud_count"] == 0
    assert rows["tx2"]["origin_total_tx_count"] == 1
    assert rows["tx2"]["origin_avg_amount"] == 10.0
    assert rows["tx2"]["origin_unique_destinations"] == 1
    assert rows["tx3"]["origin_total_tx_count"] == 2
    assert rows["tx3"]["origin_avg_amount"] == 15.0
    assert rows["tx3"]["origin_destination_pair_count"] == 1
    assert rows["tx3"]["edge_frequency"] == 1
    assert rows["tx3"]["destination_historical_fraud_count"] == 0
    assert rows["tx4"]["destination_in_degree"] == 1
