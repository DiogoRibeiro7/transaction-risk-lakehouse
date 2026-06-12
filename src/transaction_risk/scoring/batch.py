"""Batch scoring for materialized transaction and feature tables."""

from __future__ import annotations

import logging
from pathlib import Path

from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from transaction_risk.features.pipeline import build_feature_table
from transaction_risk.models.evaluation import add_positive_probability
from transaction_risk.models.thresholding import add_alert_flag
from transaction_risk.spark.io import read_table, write_table

logger = logging.getLogger(__name__)


def score_features_dataframe(
    features_df: DataFrame,
    model: PipelineModel,
    threshold: float = 0.5,
) -> DataFrame:
    """Score a feature DataFrame and add alert flag and threshold columns.

    Spark ML artifact columns added by the model pipeline (imputed inputs,
    indexed categories, and vector columns) are dropped so the output stays
    readable and Parquet-friendly.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    input_columns = set(features_df.columns)
    scored = add_positive_probability(model.transform(features_df))
    artifact_columns = [
        column
        for column in scored.columns
        if column not in input_columns and column != "fraud_probability"
    ]
    cleaned = scored.drop(*artifact_columns)
    alerted = add_alert_flag(cleaned, threshold=threshold)
    return alerted.withColumn("selected_threshold", F.lit(float(threshold)))


def score_feature_table(
    spark: SparkSession,
    input_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    threshold: float = 0.5,
    table_format: str = "parquet",
) -> DataFrame:
    """Score an already materialized gold feature table with a saved model."""
    logger.info("Scoring feature table %s with model %s", input_path, model_path)
    features = read_table(spark, input_path, table_format=table_format)
    model = PipelineModel.load(str(model_path))
    scored = score_features_dataframe(features, model, threshold=threshold)
    write_table(scored, output_path, table_format=table_format)
    logger.info("Wrote scored output to %s", output_path)
    return scored


def score_transaction_table(
    spark: SparkSession,
    input_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    threshold: float = 0.5,
    table_format: str = "parquet",
) -> DataFrame:
    """Score a cleaned silver transaction table by building features first."""
    logger.info("Scoring transaction table %s with model %s", input_path, model_path)
    transactions = read_table(spark, input_path, table_format=table_format)
    features = build_feature_table(transactions)
    model = PipelineModel.load(str(model_path))
    scored = score_features_dataframe(features, model, threshold=threshold)
    write_table(scored, output_path, table_format=table_format)
    logger.info("Wrote scored output to %s", output_path)
    return scored
