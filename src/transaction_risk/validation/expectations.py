"""Lightweight data expectation checks for transaction tables.

Inspired by Great Expectations, but implemented with plain PySpark so the base
project does not depend on the Great Expectations runtime. Checks return
structured results instead of raising, so ingestion can record quality reports
without stopping the pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpectationResult:
    """Outcome of a single expectation check."""

    name: str
    success: bool
    details: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExpectationSuiteResult:
    """Outcome of a full expectation suite run."""

    dataset_name: str
    results: list[ExpectationResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Whether every expectation in the suite passed."""
        return all(result.success for result in self.results)

    @property
    def failure_count(self) -> int:
        """Number of failed expectations."""
        return sum(1 for result in self.results if not result.success)


def _missing_column_result(name: str, column: str) -> ExpectationResult:
    return ExpectationResult(
        name=name,
        success=False,
        details={"column": column, "reason": "column is missing"},
    )


def expect_columns_exist(df: DataFrame, columns: list[str]) -> ExpectationResult:
    """Check that all required columns exist."""
    missing = sorted(set(columns) - set(df.columns))
    return ExpectationResult(
        name="columns_exist",
        success=not missing,
        details={"required": ", ".join(columns), "missing": ", ".join(missing)},
    )


def expect_column_completeness(
    df: DataFrame,
    column: str,
    min_ratio: float = 0.99,
) -> ExpectationResult:
    """Check that the non-null ratio of a column meets a minimum threshold."""
    name = f"completeness_{column}"
    if column not in df.columns:
        return _missing_column_result(name, column)

    total = df.count()
    non_null = df.filter(F.col(column).isNotNull()).count()
    ratio = non_null / total if total else 1.0
    return ExpectationResult(
        name=name,
        success=ratio >= min_ratio,
        details={"column": column, "non_null_ratio": ratio, "min_ratio": min_ratio},
    )


def expect_binary_label(df: DataFrame, label_column: str = "isFraud") -> ExpectationResult:
    """Check that a label column contains only 0 and 1 values."""
    name = f"binary_label_{label_column}"
    if label_column not in df.columns:
        return _missing_column_result(name, label_column)

    invalid = df.filter(~F.col(label_column).isin(0, 1) | F.col(label_column).isNull()).count()
    return ExpectationResult(
        name=name,
        success=invalid == 0,
        details={"column": label_column, "invalid_count": invalid},
    )


def expect_non_negative(df: DataFrame, column: str) -> ExpectationResult:
    """Check that a numeric column contains no negative values."""
    name = f"non_negative_{column}"
    if column not in df.columns:
        return _missing_column_result(name, column)

    negative = df.filter(F.col(column) < 0).count()
    return ExpectationResult(
        name=name,
        success=negative == 0,
        details={"column": column, "negative_count": negative},
    )


def expect_duplicate_rate_below(
    df: DataFrame,
    key_columns: list[str],
    max_ratio: float = 0.01,
) -> ExpectationResult:
    """Check that the duplicated-row ratio over key columns stays below a threshold."""
    name = "duplicate_rate"
    missing = sorted(set(key_columns) - set(df.columns))
    if missing:
        return _missing_column_result(name, ", ".join(missing))

    total = df.count()
    distinct = df.select(*key_columns).distinct().count()
    duplicate_ratio = (total - distinct) / total if total else 0.0
    return ExpectationResult(
        name=name,
        success=duplicate_ratio <= max_ratio,
        details={
            "key_columns": ", ".join(key_columns),
            "duplicate_ratio": duplicate_ratio,
            "max_ratio": max_ratio,
        },
    )


def run_transaction_expectation_suite(
    df: DataFrame,
    dataset_name: str = "transactions",
) -> ExpectationSuiteResult:
    """Run the standard transaction expectation suite."""
    duplicate_keys = ["transaction_id"] if "transaction_id" in df.columns else df.columns

    results = [
        expect_columns_exist(df, ["step", "type", "amount", "isFraud"]),
        expect_column_completeness(df, "amount"),
        expect_column_completeness(df, "type"),
        expect_column_completeness(df, "step"),
        expect_binary_label(df, "isFraud"),
        expect_non_negative(df, "amount"),
        expect_non_negative(df, "step"),
        expect_duplicate_rate_below(df, duplicate_keys),
    ]
    suite = ExpectationSuiteResult(dataset_name=dataset_name, results=results)
    logger.info(
        "Expectation suite %s finished: %d checks, %d failures",
        dataset_name,
        len(suite.results),
        suite.failure_count,
    )
    return suite


def expectation_suite_to_json(suite: ExpectationSuiteResult) -> str:
    """Serialize a suite result as formatted JSON."""
    payload = {
        "dataset_name": suite.dataset_name,
        "success": suite.success,
        "failure_count": suite.failure_count,
        "results": [asdict(result) for result in suite.results],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def expectation_suite_to_markdown(suite: ExpectationSuiteResult) -> str:
    """Render a suite result as GitHub Markdown."""
    status = "PASSED" if suite.success else f"FAILED ({suite.failure_count} failures)"
    lines = [
        f"# Data quality report — {suite.dataset_name}",
        "",
        f"Overall status: **{status}**",
        "",
        "| check | status | details |",
        "| --- | --- | --- |",
    ]
    for result in suite.results:
        details = "; ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in result.details.items()
            if value != ""
        )
        lines.append(f"| {result.name} | {'pass' if result.success else 'fail'} | {details} |")
    return "\n".join(lines) + "\n"


def write_expectation_report(suite: ExpectationSuiteResult, output_path: str | Path) -> None:
    """Write a suite result as JSON or Markdown depending on the file suffix."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".md", ".markdown"}:
        path.write_text(expectation_suite_to_markdown(suite), encoding="utf-8")
    else:
        path.write_text(expectation_suite_to_json(suite), encoding="utf-8")
    logger.info("Wrote expectation report to %s", path)
