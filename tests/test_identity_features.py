from __future__ import annotations

from pyspark.sql.types import StringType, StructField, StructType

from transaction_risk.features.identity_features import add_identity_features
from transaction_risk.features.pipeline import build_feature_table


def test_add_identity_features_with_all_expected_columns(spark) -> None:
    df = spark.createDataFrame(
        [
            ("tx1", 1.0, "W", "gmail.com", "gmail.com", "desktop", "ios", "1111", "2222"),
            ("tx2", 3.0, "W", "yahoo.com", "hotmail.com", "mobile", None, None, "3333"),
        ],
        [
            "TransactionID",
            "TransactionAmt",
            "ProductCD",
            "P_emaildomain",
            "R_emaildomain",
            "DeviceType",
            "DeviceInfo",
            "card1",
            "card2",
        ],
    )

    result = add_identity_features(df)
    rows = {row["TransactionID"]: row.asDict() for row in result.collect()}

    assert "has_card1" in result.columns
    assert "has_card2" in result.columns
    assert "p_emaildomain_normalized" in result.columns
    assert "r_emaildomain_normalized" in result.columns
    assert "device_type_normalized" in result.columns
    assert "device_info_normalized" in result.columns
    assert "identity_missing_value_count" in result.columns
    assert rows["tx1"]["email_domains_match"] == 1
    assert rows["tx2"]["email_domains_match"] == 0
    assert rows["tx1"]["has_card1"] == 1
    assert rows["tx2"]["has_card1"] == 0


def test_add_identity_features_tolerates_missing_optional_columns(spark) -> None:
    df = spark.createDataFrame(
        [("tx1", 1.0, "W", "1111")],
        ["TransactionID", "TransactionAmt", "ProductCD", "card1"],
    )

    result = add_identity_features(df)

    assert "has_card1" in result.columns
    assert "p_emaildomain_normalized" not in result.columns
    assert "device_type_normalized" not in result.columns
    assert "product_cd_avg_transaction_amount" not in result.columns
    assert result.first()["TransactionID"] == "tx1"


def test_add_identity_features_handles_nulls(spark) -> None:
    df = spark.createDataFrame(
        [("tx1", None, None, None, None, None)],
        StructType(
            [
                StructField("TransactionID", StringType(), nullable=False),
                StructField("card1", StringType(), nullable=True),
                StructField("P_emaildomain", StringType(), nullable=True),
                StructField("R_emaildomain", StringType(), nullable=True),
                StructField("DeviceType", StringType(), nullable=True),
                StructField("ProductCD", StringType(), nullable=True),
            ]
        ),
    )

    result = add_identity_features(df).collect()[0].asDict()

    assert result["has_card1"] == 0
    assert result["has_p_emaildomain"] == 0
    assert result["has_r_emaildomain"] == 0
    assert result["email_domains_match"] == 0
    assert result["device_type_normalized"] == "unknown"
    assert result["identity_missing_value_count"] >= 4


def test_build_feature_table_keeps_identity_features_disabled_by_default(spark) -> None:
    df = spark.createDataFrame(
        [
            (0, "TRANSFER", 100.0, "C1", 100.0, 0.0, "C2", 0.0, 100.0, 1, 0, "tx1", "gmail.com"),
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
            "P_emaildomain",
        ],
    )

    result = build_feature_table(df)

    assert "p_emaildomain_normalized" not in result.columns


def test_add_identity_features_uses_historical_product_stats_when_time_exists(spark) -> None:
    df = spark.createDataFrame(
        [
            ("tx1", 1.0, "W", 1),
            ("tx2", 3.0, "W", 2),
            ("tx3", 5.0, "W", 3),
        ],
        ["TransactionID", "TransactionAmt", "ProductCD", "TransactionDT"],
    )

    result = add_identity_features(df).orderBy("TransactionDT").collect()

    assert result[0]["product_cd_transaction_count"] == 0
    assert result[0]["product_cd_avg_transaction_amount"] == 0.0
    assert result[1]["product_cd_transaction_count"] == 1
    assert result[1]["product_cd_avg_transaction_amount"] == 1.0
    assert result[2]["product_cd_transaction_count"] == 2
    assert result[2]["product_cd_max_transaction_amount"] == 3.0
