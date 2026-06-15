"""Feature pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame

from transaction_risk.features.entity_features import add_entity_features
from transaction_risk.features.graph_features import add_graph_features
from transaction_risk.features.history import resolve_time_column
from transaction_risk.features.identity_features import add_identity_features
from transaction_risk.features.temporal_features import add_temporal_features
from transaction_risk.features.transaction_features import add_transaction_features
from transaction_risk.spark.session import load_yaml_config


def _identity_features_enabled(feature_config_path: str | Path) -> bool:
    """Read the feature-config flag for optional IEEE-CIS identity features."""
    try:
        config = load_yaml_config(feature_config_path)
    except FileNotFoundError:
        return False
    return bool(config.get("enable_identity_features", False))


def build_feature_table(
    df: DataFrame,
    enable_identity_features: bool | None = None,
    feature_config_path: str | Path = "conf/features.yaml",
) -> DataFrame:
    """Build the full model-ready feature table from silver transactions."""
    time_column = resolve_time_column(df)
    featured = add_transaction_features(df)
    featured = add_temporal_features(featured, time_column=time_column)
    featured = add_entity_features(featured, time_column=time_column)
    featured = add_graph_features(featured, time_column=time_column)
    identity_enabled = (
        enable_identity_features
        if enable_identity_features is not None
        else _identity_features_enabled(feature_config_path)
    )
    if identity_enabled:
        featured = add_identity_features(featured, time_column=time_column)
    return featured
