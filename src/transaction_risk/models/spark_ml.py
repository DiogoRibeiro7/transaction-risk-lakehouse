"""Spark ML model pipelines for fraud risk scoring."""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from transaction_risk.validation.schema import require_columns

DEFAULT_NUMERIC_FEATURES = [
    "amount",
    "amount_log1p",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "origin_balance_delta",
    "destination_balance_delta",
    "origin_delta_minus_amount",
    "destination_delta_minus_amount",
    "origin_balance_was_zero",
    "origin_balance_is_zero",
    "destination_balance_was_zero",
    "destination_balance_is_zero",
    "is_cash_out",
    "is_transfer",
    "is_payment",
    "is_debit",
    "is_cash_in",
    "is_merchant_destination",
    "large_amount_flag",
    "steps_since_previous_origin_tx",
    "origin_tx_count_before",
    "origin_amount_mean_before",
    "origin_amount_std_before",
    "origin_amount_to_mean_ratio",
    "origin_amount_zscore_before",
    "origin_total_tx_count",
    "origin_avg_amount",
    "origin_max_amount",
    "origin_unique_destinations",
    "destination_total_tx_count",
    "destination_avg_amount",
    "destination_max_amount",
    "destination_unique_origins",
    "origin_destination_pair_count",
    "origin_destination_avg_amount",
    "origin_out_degree",
    "destination_in_degree",
    "edge_frequency",
    "destination_historical_fraud_rate",
    "destination_historical_fraud_count",
]


@dataclass(frozen=True)
class ModelTrainingConfig:
    """Configuration for Spark ML model training."""

    model_type: str = "logistic_regression"
    label_column: str = "isFraud"
    categorical_column: str = "type"
    features_column: str = "features"
    prediction_column: str = "prediction"
    probability_column: str = "probability"
    seed: int = 42


def _available_numeric_features(df: DataFrame, requested_features: list[str]) -> list[str]:
    """Return requested numeric features that exist in the DataFrame."""
    return [feature for feature in requested_features if feature in df.columns]


def with_class_weights(df: DataFrame, label_column: str = "isFraud", weight_column: str = "class_weight") -> DataFrame:
    """Add inverse-frequency class weights for imbalanced classification."""
    require_columns(df, [label_column])
    counts = {row[label_column]: row["count"] for row in df.groupBy(label_column).count().collect()}
    negative_count = float(counts.get(0, 0))
    positive_count = float(counts.get(1, 0))

    if negative_count == 0 or positive_count == 0:
        return df.withColumn(weight_column, F.lit(1.0))

    total = negative_count + positive_count
    negative_weight = total / (2.0 * negative_count)
    positive_weight = total / (2.0 * positive_count)

    return df.withColumn(
        weight_column,
        F.when(F.col(label_column) == 1, F.lit(positive_weight)).otherwise(F.lit(negative_weight)),
    )


def build_model_pipeline(
    df: DataFrame,
    config: ModelTrainingConfig | None = None,
    numeric_features: list[str] | None = None,
) -> Pipeline:
    """Build a Spark ML pipeline for fraud-risk classification."""
    cfg = config or ModelTrainingConfig()
    require_columns(df, [cfg.label_column, cfg.categorical_column])

    feature_columns = _available_numeric_features(df, numeric_features or DEFAULT_NUMERIC_FEATURES)
    if not feature_columns:
        raise ValueError("No numeric feature columns are available for training.")

    imputed_columns = [f"{column}_imputed" for column in feature_columns]
    imputer = Imputer(inputCols=feature_columns, outputCols=imputed_columns).setStrategy("median")
    type_indexer = StringIndexer(
        inputCol=cfg.categorical_column,
        outputCol="type_index",
        handleInvalid="keep",
    )

    assembler = VectorAssembler(
        inputCols=[*imputed_columns, "type_index"],
        outputCol=cfg.features_column,
        handleInvalid="keep",
    )

    model_type = cfg.model_type.lower()
    if model_type == "logistic_regression":
        estimator = LogisticRegression(
            labelCol=cfg.label_column,
            featuresCol=cfg.features_column,
            weightCol="class_weight",
            probabilityCol=cfg.probability_column,
            predictionCol=cfg.prediction_column,
            maxIter=50,
            regParam=0.01,
        )
    elif model_type == "random_forest":
        estimator = RandomForestClassifier(
            labelCol=cfg.label_column,
            featuresCol=cfg.features_column,
            weightCol="class_weight",
            probabilityCol=cfg.probability_column,
            predictionCol=cfg.prediction_column,
            numTrees=80,
            maxDepth=8,
            seed=cfg.seed,
        )
    elif model_type == "gbt":
        estimator = GBTClassifier(
            labelCol=cfg.label_column,
            featuresCol=cfg.features_column,
            weightCol="class_weight",
            predictionCol=cfg.prediction_column,
            maxIter=60,
            maxDepth=5,
            seed=cfg.seed,
        )
    else:
        raise ValueError(f"Unsupported model_type: {cfg.model_type}")

    return Pipeline(stages=[imputer, type_indexer, assembler, estimator])


def train_model(
    train_df: DataFrame,
    config: ModelTrainingConfig | None = None,
    numeric_features: list[str] | None = None,
) -> PipelineModel:
    """Train a Spark ML fraud-risk model."""
    cfg = config or ModelTrainingConfig()
    weighted = with_class_weights(train_df, label_column=cfg.label_column)
    pipeline = build_model_pipeline(weighted, config=cfg, numeric_features=numeric_features)
    return pipeline.fit(weighted)
