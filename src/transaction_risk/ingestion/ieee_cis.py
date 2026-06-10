"""Skeleton ingestion module for the IEEE-CIS fraud detection dataset.

The dataset contains separate transaction and identity tables. The implementation is kept
lightweight on purpose because the default reproducible demo uses PaySim.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def read_ieee_cis(
    spark: SparkSession,
    transaction_path: str | Path,
    identity_path: str | Path | None = None,
) -> DataFrame:
    """Read and optionally join IEEE-CIS transaction and identity tables.

    Parameters
    ----------
    spark:
        Active Spark session.
    transaction_path:
        Path to `train_transaction.csv` or a compatible file.
    identity_path:
        Optional path to `train_identity.csv`.

    Returns
    -------
    DataFrame
        Joined transaction table. Identity fields may be null for many transactions.
    """
    transactions = spark.read.option("header", "true").option("inferSchema", "true").csv(str(transaction_path))

    if identity_path is None:
        return transactions

    identities = spark.read.option("header", "true").option("inferSchema", "true").csv(str(identity_path))
    return transactions.join(identities, on="TransactionID", how="left").withColumn(
        "has_identity", F.col("id_01").isNotNull().cast("int")
    )
