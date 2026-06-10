"""PaySim ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from transaction_risk.spark.io import read_csv, write_table
from transaction_risk.validation.schema import require_columns

PAYSIM_SCHEMA = StructType(
    [
        StructField("step", IntegerType(), nullable=False),
        StructField("type", StringType(), nullable=False),
        StructField("amount", DoubleType(), nullable=False),
        StructField("nameOrig", StringType(), nullable=False),
        StructField("oldbalanceOrg", DoubleType(), nullable=False),
        StructField("newbalanceOrig", DoubleType(), nullable=False),
        StructField("nameDest", StringType(), nullable=False),
        StructField("oldbalanceDest", DoubleType(), nullable=False),
        StructField("newbalanceDest", DoubleType(), nullable=False),
        StructField("isFraud", IntegerType(), nullable=False),
        StructField("isFlaggedFraud", IntegerType(), nullable=False),
    ]
)

REQUIRED_COLUMNS = [field.name for field in PAYSIM_SCHEMA.fields]


def read_paysim_raw(spark: SparkSession, input_path: str | Path) -> DataFrame:
    """Read raw PaySim CSV data with a strict schema."""
    df = read_csv(spark=spark, path=input_path, schema=PAYSIM_SCHEMA, header=True)
    require_columns(df, REQUIRED_COLUMNS)
    return df


def clean_paysim_transactions(df: DataFrame) -> DataFrame:
    """Clean and normalize PaySim transactions.

    The PaySim dataset does not include a transaction identifier. This function creates a
    deterministic `transaction_id` from the ordered transaction fields and removes duplicate rows.
    """
    require_columns(df, REQUIRED_COLUMNS)

    cleaned = (
        df.dropna(subset=REQUIRED_COLUMNS)
        .withColumn("type", F.upper(F.trim(F.col("type"))))
        .withColumn("amount", F.col("amount").cast("double"))
        .withColumn("oldbalanceOrg", F.col("oldbalanceOrg").cast("double"))
        .withColumn("newbalanceOrig", F.col("newbalanceOrig").cast("double"))
        .withColumn("oldbalanceDest", F.col("oldbalanceDest").cast("double"))
        .withColumn("newbalanceDest", F.col("newbalanceDest").cast("double"))
        .withColumn("isFraud", F.col("isFraud").cast("int"))
        .withColumn("isFlaggedFraud", F.col("isFlaggedFraud").cast("int"))
        .filter(F.col("amount") >= 0)
        .filter(F.col("step") >= 0)
    )

    key_columns = [
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
    ]

    return (
        cleaned.withColumn(
            "transaction_id",
            F.sha2(F.concat_ws("||", *[F.col(column).cast("string") for column in key_columns]), 256),
        )
        .dropDuplicates(["transaction_id"])
        .select("transaction_id", *REQUIRED_COLUMNS)
    )


def ingest_paysim(
    spark: SparkSession,
    input_path: str | Path,
    bronze_path: str | Path,
    silver_path: str | Path,
    table_format: str = "parquet",
) -> tuple[DataFrame, DataFrame]:
    """Run the PaySim ingestion pipeline.

    Returns the bronze and silver DataFrames for optional immediate downstream use.
    """
    bronze = read_paysim_raw(spark, input_path)
    write_table(bronze, bronze_path, table_format=table_format)

    silver = clean_paysim_transactions(bronze)
    write_table(silver, silver_path, table_format=table_format)
    return bronze, silver
