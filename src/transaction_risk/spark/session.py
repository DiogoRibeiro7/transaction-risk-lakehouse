"""Spark session utilities."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def apply_spark_java_home() -> None:
    """Point Spark at a dedicated JDK when `SPARK_JAVA_HOME` is set.

    Spark 4 requires Java 17 or 21. Machines whose default `JAVA_HOME` is a
    newer JDK (Java 24 removed the Security Manager that Hadoop still needs)
    can set `SPARK_JAVA_HOME` to a compatible JDK without changing the
    machine-wide Java installation.
    """
    spark_java_home = os.environ.get("SPARK_JAVA_HOME")
    if spark_java_home and os.environ.get("JAVA_HOME") != spark_java_home:
        logger.info("Using SPARK_JAVA_HOME for the Spark JVM: %s", spark_java_home)
        os.environ["JAVA_HOME"] = spark_java_home


def apply_hadoop_home() -> None:
    """Expose `HADOOP_HOME/bin` on PATH so Hadoop native binaries are found.

    On Windows, Hadoop filesystem operations (including Spark ML model saves)
    require `winutils.exe` and `hadoop.dll`. Set `HADOOP_HOME` to a folder with
    a `bin` directory containing them.
    """
    hadoop_home = os.environ.get("HADOOP_HOME")
    if not hadoop_home:
        return
    hadoop_bin = str(Path(hadoop_home) / "bin")
    current_path = os.environ.get("PATH", "")
    if hadoop_bin not in current_path.split(os.pathsep):
        os.environ["PATH"] = hadoop_bin + os.pathsep + current_path


def prepare_spark_environment() -> None:
    """Apply all local environment overrides needed before launching the JVM."""
    apply_spark_java_home()
    apply_hadoop_home()


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to a YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML content. Empty files return an empty dictionary.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Configuration file must contain a mapping: {config_path}")
    return loaded


def create_spark_session(
    app_name: str = "transaction-risk-lakehouse",
    master: str | None = "local[*]",
    configs: Mapping[str, str] | None = None,
    log_level: str = "WARN",
) -> SparkSession:
    """Create a SparkSession with safe local defaults.

    Parameters
    ----------
    app_name:
        Spark application name.
    master:
        Spark master URL. Use `None` when submitting to a managed cluster.
    configs:
        Extra Spark configuration key-value pairs.
    log_level:
        Spark context log level.

    Returns
    -------
    SparkSession
        Configured Spark session.
    """
    prepare_spark_environment()
    builder = SparkSession.builder.appName(app_name)
    if master is not None:
        builder = builder.master(master)

    for key, value in (configs or {}).items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(log_level)
    return spark


def create_spark_session_from_yaml(path: str | Path) -> SparkSession:
    """Create a SparkSession from a YAML configuration file."""
    config = load_yaml_config(path)
    app_name = str(config.get("app_name", "transaction-risk-lakehouse"))
    master = config.get("master", "local[*]")
    log_level = str(config.get("log_level", "WARN"))
    spark_configs = config.get("configs", {})

    if spark_configs is not None and not isinstance(spark_configs, dict):
        raise TypeError("The `configs` section must be a mapping.")

    return create_spark_session(
        app_name=app_name,
        master=str(master) if master is not None else None,
        configs={str(k): str(v) for k, v in (spark_configs or {}).items()},
        log_level=log_level,
    )
