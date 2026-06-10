"""Input/output helpers for Spark DataFrames."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

SUPPORTED_TABLE_FORMATS = {"parquet", "delta"}
TableFormat = Literal["parquet", "delta"]


def _normalize_table_format(table_format: str) -> TableFormat:
    """Validate and normalize a lakehouse table format name."""
    normalized = table_format.strip().lower()
    if normalized not in SUPPORTED_TABLE_FORMATS:
        supported_formats = ", ".join(sorted(SUPPORTED_TABLE_FORMATS))
        raise ValueError(
            f"Unsupported table format '{table_format}'. Supported formats: {supported_formats}."
        )
    return normalized  # type: ignore[return-value]


def _validate_partition_columns(
    df: DataFrame,
    partition_columns: list[str] | None,
) -> list[str]:
    """Validate partition columns against the DataFrame schema."""
    if partition_columns is None:
        return []
    if not partition_columns:
        raise ValueError("partition_columns must contain at least one column when provided.")

    missing_columns = sorted(set(partition_columns) - set(df.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Partition columns not present in DataFrame: {missing}.")
    return partition_columns


def read_csv(
    spark: SparkSession,
    path: str | Path,
    schema: StructType | None = None,
    header: bool = True,
) -> DataFrame:
    """Read a CSV file with optional schema enforcement."""
    reader = spark.read.option("header", str(header).lower()).option("mode", "FAILFAST")
    reader = reader.schema(schema) if schema is not None else reader.option("inferSchema", "true")
    return reader.csv(str(path))


def read_parquet(spark: SparkSession, path: str | Path) -> DataFrame:
    """Read a Parquet dataset."""
    return read_table(spark=spark, path=path, table_format="parquet")


def write_parquet(df: DataFrame, path: str | Path, mode: str = "overwrite") -> None:
    """Write a DataFrame as Parquet."""
    write_table(df=df, path=path, table_format="parquet", mode=mode)


def write_partitioned_parquet(
    df: DataFrame,
    path: str | Path,
    partition_columns: list[str],
    mode: str = "overwrite",
) -> None:
    """Write a partitioned Parquet dataset."""
    write_table(
        df=df,
        path=path,
        table_format="parquet",
        partition_columns=partition_columns,
        mode=mode,
    )


def read_table(
    spark: SparkSession,
    path: str | Path,
    table_format: str = "parquet",
) -> DataFrame:
    """Read a Parquet or Delta dataset."""
    normalized_format = _normalize_table_format(table_format)
    try:
        return spark.read.format(normalized_format).load(str(path))
    except Exception as exc:
        if normalized_format == "delta":
            raise ValueError(
                "Delta Lake support is not available in this Spark environment. "
                "Install Delta Lake dependencies or use --table-format parquet."
            ) from exc
        raise


def write_table(
    df: DataFrame,
    path: str | Path,
    table_format: str = "parquet",
    mode: str = "overwrite",
    partition_columns: list[str] | None = None,
) -> None:
    """Write a Parquet or Delta dataset, optionally partitioned."""
    normalized_format = _normalize_table_format(table_format)
    writer = df.write.mode(mode)
    validated_partition_columns = _validate_partition_columns(df, partition_columns)
    if validated_partition_columns:
        writer = writer.partitionBy(*validated_partition_columns)

    try:
        writer.format(normalized_format).save(str(path))
    except Exception as exc:
        if normalized_format == "delta":
            raise ValueError(
                "Delta Lake support is not available in this Spark environment. "
                "Install Delta Lake dependencies or use --table-format parquet."
            ) from exc
        raise
