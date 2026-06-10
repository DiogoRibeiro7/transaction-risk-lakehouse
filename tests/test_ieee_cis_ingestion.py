from __future__ import annotations

import csv
from pathlib import Path

import pytest

from transaction_risk.ingestion.ieee_cis import ingest_ieee_cis, read_ieee_cis

FIXTURES_DIR = Path("tests") / "fixtures" / "ieee_cis"


def _csv_fixture_to_df(spark, path: Path):
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return spark.createDataFrame(rows)


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

    assert result.count() == 3
    rows = {row["TransactionID"]: row.asDict() for row in result.collect()}
    assert rows["1003"]["DeviceType"] is None
    assert rows["1003"]["has_identity"] == 0
    assert rows["1002"]["has_identity"] == 1
    assert rows["1001"]["isFraud"] == 0
    assert rows["1002"]["isFraud"] == 1
    assert rows["1001"]["event_time"] is not None


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
    writes: list[tuple[str, int]] = []

    def capture_write(df, path, table_format="parquet", mode="overwrite", partition_columns=None) -> None:
        assert table_format == "parquet"
        assert mode == "overwrite"
        assert partition_columns is None
        writes.append((str(path), df.count()))

    monkeypatch.setattr("transaction_risk.ingestion.ieee_cis.write_table", capture_write)

    transactions, identities, silver = ingest_ieee_cis(
        spark=spark,
        transaction_path=FIXTURES_DIR / "train_transaction.csv",
        identity_path=FIXTURES_DIR / "train_identity.csv",
        bronze_path=bronze_path,
        silver_path=silver_path,
    )

    assert transactions.count() == 3
    assert identities.count() == 2
    assert silver.count() == 3
    assert writes == [
        (str(bronze_path / "transactions"), 3),
        (str(bronze_path / "identity"), 2),
        (str(silver_path), 3),
    ]


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
