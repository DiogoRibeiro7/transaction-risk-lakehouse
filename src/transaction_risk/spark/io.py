"""Input/output helpers for Spark DataFrames."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType


def read_csv(
    spark: SparkSession,
    path: str | Path,
    schema: StructType | None = None,
    header: bool = True,
) -> DataFrame:
    """Read a CSV file with optional schema enforcement."""
    reader = spark.read.option("header", str(header).lower()).option("mode", "FAILFAST")
    if schema is not None:
        reader = reader.schema(schema)
    else:
        reader = reader.option("inferSchema", "true")
    return reader.csv(str(path))


def read_parquet(spark: SparkSession, path: str | Path) -> DataFrame:
    """Read a Parquet dataset."""
    return spark.read.parquet(str(path))


def write_parquet(df: DataFrame, path: str | Path, mode: str = "overwrite") -> None:
    """Write a DataFrame as Parquet."""
    df.write.mode(mode).parquet(str(path))


def write_partitioned_parquet(
    df: DataFrame,
    path: str | Path,
    partition_columns: list[str],
    mode: str = "overwrite",
) -> None:
    """Write a partitioned Parquet dataset."""
    if not partition_columns:
        raise ValueError("partition_columns must contain at least one column.")
    df.write.mode(mode).partitionBy(*partition_columns).parquet(str(path))
