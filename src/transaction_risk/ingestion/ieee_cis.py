"""IEEE-CIS ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType

from transaction_risk.spark.io import read_csv, write_table
from transaction_risk.validation.data_quality import duplicate_count
from transaction_risk.validation.schema import require_binary_label, require_columns

TRANSACTION_REQUIRED_COLUMNS = ["TransactionID", "TransactionDT", "TransactionAmt"]
IDENTITY_REQUIRED_COLUMNS = ["TransactionID"]
OPTIONAL_LABEL_COLUMNS = ["isFraud"]
BRONZE_TRANSACTION_TABLE_NAME = "transactions"
BRONZE_IDENTITY_TABLE_NAME = "identity"


def _normalize_transaction_schema(df: DataFrame) -> DataFrame:
    """Normalize core IEEE-CIS transaction columns to stable types."""
    require_columns(df, TRANSACTION_REQUIRED_COLUMNS)

    normalized = (
        df.withColumn("TransactionID", F.col("TransactionID").cast(StringType()))
        .withColumn("TransactionDT", F.col("TransactionDT").cast(IntegerType()))
        .withColumn("TransactionAmt", F.col("TransactionAmt").cast(DoubleType()))
    )

    if "isFraud" in normalized.columns:
        normalized = normalized.withColumn("isFraud", F.col("isFraud").cast(IntegerType()))
    return normalized


def _normalize_identity_schema(df: DataFrame) -> DataFrame:
    """Normalize core IEEE-CIS identity columns to stable types."""
    require_columns(df, IDENTITY_REQUIRED_COLUMNS)
    return df.withColumn("TransactionID", F.col("TransactionID").cast(StringType()))


def _read_transaction_csv(spark: SparkSession, path: str | Path) -> DataFrame:
    """Read a transaction CSV and normalize its schema."""
    return _normalize_transaction_schema(read_csv(spark, path, header=True))


def _read_identity_csv(spark: SparkSession, path: str | Path) -> DataFrame:
    """Read an identity CSV and normalize its schema."""
    return _normalize_identity_schema(read_csv(spark, path, header=True))


def _validate_ieee_transactions(df: DataFrame) -> None:
    """Run ingestion-time quality checks for IEEE-CIS transactions."""
    require_columns(df, TRANSACTION_REQUIRED_COLUMNS)

    missing_transaction_id_count = df.filter(F.col("TransactionID").isNull()).count()
    if missing_transaction_id_count > 0:
        raise ValueError("IEEE-CIS transactions contain missing TransactionID values.")

    duplicate_transaction_ids = duplicate_count(df, ["TransactionID"])
    if duplicate_transaction_ids > 0:
        raise ValueError(
            "IEEE-CIS transactions contain duplicate TransactionID values."
        )

    if "isFraud" in df.columns:
        require_binary_label(df, "isFraud")


def _validate_ieee_identities(df: DataFrame) -> None:
    """Run ingestion-time quality checks for IEEE-CIS identities."""
    require_columns(df, IDENTITY_REQUIRED_COLUMNS)

    missing_transaction_id_count = df.filter(F.col("TransactionID").isNull()).count()
    if missing_transaction_id_count > 0:
        raise ValueError("IEEE-CIS identities contain missing TransactionID values.")

    duplicate_transaction_ids = duplicate_count(df, ["TransactionID"])
    if duplicate_transaction_ids > 0:
        raise ValueError("IEEE-CIS identities contain duplicate TransactionID values.")


def _add_ieee_identity_columns(
    transactions: DataFrame,
    identities: DataFrame | None = None,
) -> DataFrame:
    """Join IEEE-CIS identities and derived flags onto transaction rows."""
    if identities is None:
        return transactions.withColumn("event_time", F.to_timestamp(F.from_unixtime("TransactionDT")))

    joined = transactions.join(identities, on="TransactionID", how="left")
    identity_columns = [column for column in identities.columns if column != "TransactionID"]

    has_identity_expression = F.lit(0)
    if identity_columns:
        has_identity_expression = F.greatest(
            *[F.col(column).isNotNull().cast("int") for column in identity_columns]
        )

    return joined.withColumn("has_identity", has_identity_expression.cast("int")).withColumn(
        "event_time",
        F.to_timestamp(F.from_unixtime("TransactionDT")),
    )


def read_ieee_cis(
    spark: SparkSession,
    transaction_path: str | Path,
    identity_path: str | Path | None = None,
) -> DataFrame:
    """Read and optionally join IEEE-CIS transaction and identity tables."""
    transactions = _read_transaction_csv(spark, transaction_path)
    _validate_ieee_transactions(transactions)

    if identity_path is None:
        return _add_ieee_identity_columns(transactions)

    identities = _read_identity_csv(spark, identity_path)
    _validate_ieee_identities(identities)
    return _add_ieee_identity_columns(transactions, identities)


def ingest_ieee_cis(
    spark: SparkSession,
    transaction_path: str | Path,
    identity_path: str | Path,
    bronze_path: str | Path,
    silver_path: str | Path,
    table_format: str = "parquet",
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Ingest IEEE-CIS transaction and identity CSV files into bronze and silver tables."""
    transactions = _read_transaction_csv(spark, transaction_path)
    identities = _read_identity_csv(spark, identity_path)
    _validate_ieee_transactions(transactions)
    _validate_ieee_identities(identities)

    bronze_root = Path(bronze_path)
    write_table(
        transactions,
        bronze_root / BRONZE_TRANSACTION_TABLE_NAME,
        table_format=table_format,
    )
    write_table(
        identities,
        bronze_root / BRONZE_IDENTITY_TABLE_NAME,
        table_format=table_format,
    )

    silver = _add_ieee_identity_columns(transactions, identities)
    write_table(silver, silver_path, table_format=table_format)
    return transactions, identities, silver
