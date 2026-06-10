from __future__ import annotations

import pytest

from transaction_risk.validation.schema import require_columns


def test_require_columns_accepts_existing_columns(spark) -> None:
    df = spark.createDataFrame([(1, "a")], ["id", "value"])
    require_columns(df, ["id", "value"])


def test_require_columns_raises_for_missing_columns(spark) -> None:
    df = spark.createDataFrame([(1, "a")], ["id", "value"])
    with pytest.raises(ValueError, match="Missing required columns"):
        require_columns(df, ["id", "missing"])
