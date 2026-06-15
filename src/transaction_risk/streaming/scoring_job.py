"""Structured Streaming scoring job."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from transaction_risk.features.pipeline import build_feature_table
from transaction_risk.ingestion.paysim import PAYSIM_SCHEMA
from transaction_risk.models.artifact import load_scoring_artifact
from transaction_risk.scoring.batch import score_features_dataframe


def run_file_stream_scoring(
    spark: SparkSession,
    input_stream_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    checkpoint_path: str | Path,
    threshold: float = 0.5,
) -> None:
    """Score incoming PaySim-like CSV files with a saved Spark ML pipeline model.

    The job uses `foreachBatch` so the regular batch feature pipeline can be reused
    inside each micro-batch. This keeps the demo close to production code while
    avoiding unsupported unbounded aggregations in continuous streaming mode.
    """
    artifact = load_scoring_artifact(model_path)
    stream_df = (
        spark.readStream.schema(PAYSIM_SCHEMA)
        .option("header", "true")
        .csv(str(input_stream_path))
    )

    def score_micro_batch(batch_df: DataFrame, batch_id: int) -> None:
        # DataFrame.isEmpty() stays on the JVM; batch_df.rdd.isEmpty() would force a
        # Python worker per micro-batch, which is slower and fragile on some platforms.
        if batch_df.isEmpty():
            return
        features = build_feature_table(batch_df)
        alerts = score_features_dataframe(features, artifact, threshold=threshold).withColumn(
            "stream_batch_id",
            F.lit(batch_id),
        )
        alerts.write.mode("append").parquet(str(output_path))

    query = (
        stream_df.writeStream.foreachBatch(score_micro_batch)
        .option("checkpointLocation", str(checkpoint_path))
        .outputMode("append")
        .start()
    )
    query.awaitTermination()
