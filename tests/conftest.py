"""Pytest fixtures for Spark tests."""

from __future__ import annotations

import os
import sys

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Create a local Spark session for tests."""
    python_executable = sys.executable
    os.environ.setdefault("PYSPARK_PYTHON", python_executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", python_executable)
    session = (
        SparkSession.builder.appName("transaction-risk-tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.pyspark.python", python_executable)
        .config("spark.pyspark.driver.python", python_executable)
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
