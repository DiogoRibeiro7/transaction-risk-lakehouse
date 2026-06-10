from __future__ import annotations

import pytest

from transaction_risk.spark.io import read_table, write_table


class StubReader:
    def __init__(self) -> None:
        self.selected_format: str | None = None
        self.loaded_path: str | None = None

    def format(self, value: str) -> StubReader:
        self.selected_format = value
        return self

    def load(self, path: str) -> dict[str, str]:
        self.loaded_path = path
        return {"format": self.selected_format or "", "path": self.loaded_path}


class StubSpark:
    def __init__(self) -> None:
        self.read = StubReader()


class StubWriter:
    def __init__(self) -> None:
        self.selected_mode: str | None = None
        self.partition_columns: tuple[str, ...] = ()
        self.selected_format: str | None = None
        self.saved_path: str | None = None

    def mode(self, value: str) -> StubWriter:
        self.selected_mode = value
        return self

    def partitionBy(self, *columns: str) -> StubWriter:
        self.partition_columns = columns
        return self

    def format(self, value: str) -> StubWriter:
        self.selected_format = value
        return self

    def save(self, path: str) -> None:
        self.saved_path = path


class StubDataFrame:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.write = StubWriter()


def test_write_table_rejects_unsupported_format() -> None:
    df = StubDataFrame(columns=["id", "value"])

    with pytest.raises(ValueError, match="Unsupported table format"):
        write_table(df, "dataset", table_format="iceberg")  # type: ignore[arg-type]


def test_read_table_rejects_unsupported_format() -> None:
    spark = StubSpark()

    with pytest.raises(ValueError, match="Unsupported table format"):
        read_table(spark, "dataset", table_format="iceberg")  # type: ignore[arg-type]


def test_write_table_defaults_to_parquet() -> None:
    df = StubDataFrame(columns=["id", "value"])

    write_table(df, "dataset")  # type: ignore[arg-type]

    assert df.write.selected_mode == "overwrite"
    assert df.write.selected_format == "parquet"
    assert df.write.saved_path == "dataset"


def test_write_table_validates_partition_columns() -> None:
    df = StubDataFrame(columns=["id", "value"])

    with pytest.raises(ValueError, match="partition_columns must contain at least one column"):
        write_table(df, "dataset", partition_columns=[])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Partition columns not present in DataFrame: missing"):
        write_table(df, "dataset", partition_columns=["missing"])  # type: ignore[arg-type]
