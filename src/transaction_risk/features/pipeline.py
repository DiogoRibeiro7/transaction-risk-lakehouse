"""Feature pipeline orchestration."""

from __future__ import annotations

from pyspark.sql import DataFrame

from transaction_risk.features.entity_features import add_entity_features
from transaction_risk.features.graph_features import add_graph_features
from transaction_risk.features.temporal_features import add_temporal_features
from transaction_risk.features.transaction_features import add_transaction_features


def build_feature_table(df: DataFrame) -> DataFrame:
    """Build the full model-ready feature table from silver transactions."""
    featured = add_transaction_features(df)
    featured = add_temporal_features(featured)
    featured = add_entity_features(featured)
    featured = add_graph_features(featured)
    return featured
