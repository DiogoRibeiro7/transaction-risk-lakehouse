"""Rule-based fraud risk baseline."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from transaction_risk.validation.schema import require_columns


def add_rule_based_score(df: DataFrame) -> DataFrame:
    """Add a simple interpretable fraud-risk score.

    The rule is intentionally simple. It provides a baseline that machine learning
    models should beat.
    """
    require_columns(df, ["amount", "type", "newbalanceOrig", "oldbalanceOrg"])

    return df.withColumn(
        "rule_score",
        (
            0.35 * (F.col("type").isin("TRANSFER", "CASH_OUT")).cast("double")
            + 0.25 * (F.col("newbalanceOrig") == 0).cast("double")
            + 0.25 * (F.col("amount") > 100_000).cast("double")
            + 0.15 * (F.col("amount") >= F.col("oldbalanceOrg")).cast("double")
        ),
    )
