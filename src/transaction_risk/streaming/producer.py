"""Small helper for creating local streaming input batches."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import SparkSession


def create_file_stream_batches(
    spark: SparkSession,
    source_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 500,
) -> None:
    """Split a static CSV file into small CSV files for streaming demos."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    source = Path(source_path)
    target = Path(output_dir)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    target.mkdir(parents=True, exist_ok=True)

    df = spark.read.option("header", "true").option("inferSchema", "true").csv(str(source))
    df.withColumn("batch_id", (df["step"] / batch_size).cast("int")).write.mode("overwrite").partitionBy(
        "batch_id"
    ).option("header", "true").csv(str(target))
