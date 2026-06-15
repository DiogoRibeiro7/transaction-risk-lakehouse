from __future__ import annotations

import csv
from pathlib import Path

import pytest

from transaction_risk.ingestion.ieee_cis import (
    _add_ieee_identity_columns,
    ingest_ieee_cis,
    read_ieee_cis,
)

FIXTURES_DIR = Path("tests") / "fixtures" / "ieee_cis"


def _csv_fixture_to_df(spark, path: Path):
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return spark.createDataFrame(rows)


def test_add_ieee_identity_columns_smoke(spark) -> None:
    transactions = spark.createDataFrame(
        [
            ("1001", 10, 25.5, 0),
            ("1002", 20, 99.99, 1),
            ("1003", 30, 10.0, 0),
        ],
        ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"],
    )
    identities = spark.createDataFrame(
        [
            ("1001", "desktop"),
            ("1002", "mobile"),
        ],
        ["TransactionID", "DeviceType"],
    )

    result = _add_ieee_identity_columns(transactions, identities)
    rows = {row["TransactionID"]: row.asDict() for row in result.collect()}

    assert rows["1001"]["has_identity"] == 1
    assert rows["1002"]["has_identity"] == 1
    assert rows["1003"]["has_identity"] == 0
    assert rows["1001"]["event_time"] is not None


@pytest.mark.slow
@pytest.mark.spark_slow
def test_read_ieee_cis_preserves_transactions_without_identity(
    spark,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "transaction_risk.ingestion.ieee_cis.read_csv",
        lambda spark_session, path, schema=None, header=True: _csv_fixture_to_df(spark_session, Path(path)),
    )

    result = read_ieee_cis(
        spark,
        FIXTURES_DIR / "train_transaction.csv",
        FIXTURES_DIR / "train_identity.csv",
    )

    rows = {row["TransactionID"]: row.asDict() for row in result.collect()}
    assert len(rows) == 3
    assert rows["1003"]["DeviceType"] is None
    assert rows["1003"]["has_identity"] == 0
    assert rows["1002"]["has_identity"] == 1
    assert rows["1001"]["isFraud"] == 0
    assert rows["1002"]["isFraud"] == 1
    assert rows["1001"]["event_time"] is not None


@pytest.mark.slow
@pytest.mark.spark_slow
def test_ingest_ieee_cis_writes_bronze_and_silver_tables(
    spark,
    repo_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "transaction_risk.ingestion.ieee_cis.read_csv",
        lambda spark_session, path, schema=None, header=True: _csv_fixture_to_df(spark_session, Path(path)),
    )

    bronze_path = repo_tmp_path / "bronze"
    silver_path = repo_tmp_path / "silver"
    writes: list[str] = []

    def capture_write(df, path, table_format="parquet", mode="overwrite", partition_columns=None) -> None:
        assert table_format == "parquet"
        assert mode == "overwrite"
        assert partition_columns is None
        writes.append(str(path))

    monkeypatch.setattr("transaction_risk.ingestion.ieee_cis.write_table", capture_write)

    transactions, identities, silver = ingest_ieee_cis(
        spark=spark,
        transaction_path=FIXTURES_DIR / "train_transaction.csv",
        identity_path=FIXTURES_DIR / "train_identity.csv",
        bronze_path=bronze_path,
        silver_path=silver_path,
    )

    assert len(transactions.collect()) == 3
    assert len(identities.collect()) == 2
    assert len(silver.collect()) == 3
    assert writes == [
        str(bronze_path / "transactions"),
        str(bronze_path / "identity"),
        str(silver_path),
    ]


@pytest.mark.slow
@pytest.mark.spark_slow
def test_ingest_ieee_cis_rejects_duplicate_transaction_ids(
    spark,
    repo_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "transaction_risk.ingestion.ieee_cis.read_csv",
        lambda spark_session, path, schema=None, header=True: _csv_fixture_to_df(spark_session, Path(path)),
    )

    with pytest.raises(ValueError, match="duplicate TransactionID"):
        ingest_ieee_cis(
            spark=spark,
            transaction_path=FIXTURES_DIR / "train_transaction_duplicates.csv",
            identity_path=FIXTURES_DIR / "train_identity.csv",
            bronze_path=repo_tmp_path / "bronze",
            silver_path=repo_tmp_path / "silver",
        )


@pytest.mark.slow
@pytest.mark.spark_slow
def test_ingest_ieee_cis_rejects_duplicate_identity_transaction_ids(
    spark,
    repo_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction_df = _csv_fixture_to_df(spark, FIXTURES_DIR / "train_transaction.csv")
    duplicate_identity_df = spark.createDataFrame(
        [
            {"TransactionID": "1001", "DeviceType": "desktop"},
            {"TransactionID": "1001", "DeviceType": "mobile"},
        ]
    )

    def fake_read_csv(spark_session, path, schema=None, header=True):
        if Path(path).name == "train_transaction.csv":
            return transaction_df
        return duplicate_identity_df

    monkeypatch.setattr("transaction_risk.ingestion.ieee_cis.read_csv", fake_read_csv)

    with pytest.raises(ValueError, match="identities contain duplicate TransactionID"):
        ingest_ieee_cis(
            spark=spark,
            transaction_path=FIXTURES_DIR / "train_transaction.csv",
            identity_path=FIXTURES_DIR / "train_identity.csv",
            bronze_path=repo_tmp_path / "bronze",
            silver_path=repo_tmp_path / "silver",
        )
