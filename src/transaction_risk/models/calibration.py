"""Probability calibration utilities for Spark ML fraud models."""

from __future__ import annotations

import logging

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import IsotonicRegression
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from transaction_risk.validation.schema import require_columns

logger = logging.getLogger(__name__)

CALIBRATION_FEATURES_COLUMN = "calibration_features"
CALIBRATION_PROBABILITY_COLUMN = "calibration_probability"
CALIBRATION_PREDICTION_COLUMN = "calibration_prediction"
CALIBRATION_RAW_PREDICTION_COLUMN = "calibration_raw_prediction"


def fit_platt_calibrator(
    validation_scored_df: DataFrame,
    score_column: str = "fraud_probability",
    label_column: str = "isFraud",
) -> PipelineModel:
    """Fit a Platt-scaling calibrator (logistic regression on the raw score)."""
    require_columns(validation_scored_df, [score_column, label_column])

    assembler = VectorAssembler(
        inputCols=[score_column],
        outputCol=CALIBRATION_FEATURES_COLUMN,
        handleInvalid="keep",
    )
    regression = LogisticRegression(
        featuresCol=CALIBRATION_FEATURES_COLUMN,
        labelCol=label_column,
        probabilityCol=CALIBRATION_PROBABILITY_COLUMN,
        predictionCol=CALIBRATION_PREDICTION_COLUMN,
        rawPredictionCol=CALIBRATION_RAW_PREDICTION_COLUMN,
        maxIter=100,
    )
    pipeline = Pipeline(stages=[assembler, regression])
    logger.info("Fitting Platt calibrator on score column %s", score_column)
    return pipeline.fit(validation_scored_df)


def fit_isotonic_calibrator(
    validation_scored_df: DataFrame,
    score_column: str = "fraud_probability",
    label_column: str = "isFraud",
) -> PipelineModel:
    """Fit an isotonic-regression calibrator on the raw score."""
    require_columns(validation_scored_df, [score_column, label_column])

    assembler = VectorAssembler(
        inputCols=[score_column],
        outputCol=CALIBRATION_FEATURES_COLUMN,
        handleInvalid="keep",
    )
    regression = IsotonicRegression(
        featuresCol=CALIBRATION_FEATURES_COLUMN,
        labelCol=label_column,
        predictionCol=CALIBRATION_PREDICTION_COLUMN,
        isotonic=True,
    )
    pipeline = Pipeline(stages=[assembler, regression])
    logger.info("Fitting isotonic calibrator on score column %s", score_column)
    return pipeline.fit(validation_scored_df)


def apply_calibrator(
    scored_df: DataFrame,
    calibrator_model: PipelineModel,
    output_column: str = "calibrated_fraud_probability",
) -> DataFrame:
    """Apply a fitted calibrator and add a calibrated probability column."""
    transformed = calibrator_model.transform(scored_df)

    if CALIBRATION_PROBABILITY_COLUMN in transformed.columns:
        calibrated = transformed.withColumn(
            output_column,
            vector_to_array(F.col(CALIBRATION_PROBABILITY_COLUMN))[1],
        )
    elif CALIBRATION_PREDICTION_COLUMN in transformed.columns:
        clipped = F.least(
            F.lit(1.0),
            F.greatest(F.lit(0.0), F.col(CALIBRATION_PREDICTION_COLUMN).cast("double")),
        )
        calibrated = transformed.withColumn(output_column, clipped)
    else:
        raise ValueError(
            "Calibrator output does not contain a known calibration column. "
            "Fit the calibrator with fit_platt_calibrator or fit_isotonic_calibrator."
        )

    helper_columns = [
        column
        for column in (
            CALIBRATION_FEATURES_COLUMN,
            CALIBRATION_PROBABILITY_COLUMN,
            CALIBRATION_PREDICTION_COLUMN,
            CALIBRATION_RAW_PREDICTION_COLUMN,
        )
        if column in calibrated.columns
    ]
    return calibrated.drop(*helper_columns)


def calibration_table(
    scored_df: DataFrame,
    probability_column: str,
    label_column: str = "isFraud",
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    """Bin predicted probabilities and compare them with observed fraud rates."""
    require_columns(scored_df, [probability_column, label_column])
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    bin_index = F.least(
        F.floor(F.col(probability_column) * n_bins).cast("int"),
        F.lit(n_bins - 1),
    )
    aggregated = (
        scored_df.withColumn("calibration_bin", bin_index)
        .groupBy("calibration_bin")
        .agg(
            F.count(F.lit(1)).alias("count"),
            F.avg(F.col(probability_column)).alias("mean_predicted_probability"),
            F.avg(F.col(label_column).cast("double")).alias("observed_fraud_rate"),
            F.sum(F.col(label_column).cast("int")).alias("positive_count"),
        )
        .orderBy("calibration_bin")
        .collect()
    )

    return [
        {
            "bin": int(row["calibration_bin"]),
            "bin_lower": float(row["calibration_bin"]) / n_bins,
            "bin_upper": float(row["calibration_bin"] + 1) / n_bins,
            "count": int(row["count"]),
            "positive_count": int(row["positive_count"]),
            "mean_predicted_probability": float(row["mean_predicted_probability"]),
            "observed_fraud_rate": float(row["observed_fraud_rate"]),
        }
        for row in aggregated
    ]


def brier_score(
    scored_df: DataFrame,
    probability_column: str,
    label_column: str = "isFraud",
) -> float:
    """Compute the Brier score (mean squared error between probability and label)."""
    require_columns(scored_df, [probability_column, label_column])
    squared_error = F.pow(F.col(probability_column) - F.col(label_column).cast("double"), 2)
    result = scored_df.agg(F.avg(squared_error).alias("brier_score")).collect()[0]
    return float(result["brier_score"])


def expected_calibration_error(
    scored_df: DataFrame,
    probability_column: str,
    label_column: str = "isFraud",
    n_bins: int = 10,
) -> float:
    """Approximate the expected calibration error by probability bins."""
    table = calibration_table(
        scored_df,
        probability_column=probability_column,
        label_column=label_column,
        n_bins=n_bins,
    )
    total = sum(row["count"] for row in table)
    if total == 0:
        return 0.0

    weighted_gap = sum(
        row["count"] * abs(row["mean_predicted_probability"] - row["observed_fraud_rate"])
        for row in table
    )
    return float(weighted_gap / total)
