from __future__ import annotations

import json

from transaction_risk.validation.expectations import (
    expect_binary_label,
    expect_column_completeness,
    expect_columns_exist,
    expect_duplicate_rate_below,
    expect_non_negative,
    expectation_suite_to_json,
    expectation_suite_to_markdown,
    run_transaction_expectation_suite,
)

CLEAN_COLUMNS = ["transaction_id", "step", "type", "amount", "isFraud"]


def _clean_frame(spark):
    return spark.createDataFrame(
        [
            ("t1", 1, "PAYMENT", 100.0, 0),
            ("t2", 2, "TRANSFER", 5_000.0, 1),
            ("t3", 3, "CASH_OUT", 250.0, 0),
        ],
        CLEAN_COLUMNS,
    )


def _dirty_frame(spark):
    return spark.createDataFrame(
        [
            ("t1", 1, "PAYMENT", -100.0, 0),
            ("t1", 1, "PAYMENT", -100.0, 0),
            ("t2", -2, None, 5_000.0, 2),
        ],
        CLEAN_COLUMNS,
    )


def test_suite_passes_on_clean_transactions(spark) -> None:
    suite = run_transaction_expectation_suite(_clean_frame(spark), dataset_name="clean")

    assert suite.success is True
    assert suite.failure_count == 0
    assert suite.dataset_name == "clean"
    assert len(suite.results) == 8


def test_suite_reports_failures_without_raising(spark) -> None:
    suite = run_transaction_expectation_suite(_dirty_frame(spark), dataset_name="dirty")

    assert suite.success is False
    failed = {result.name for result in suite.results if not result.success}
    assert "binary_label_isFraud" in failed
    assert "non_negative_amount" in failed
    assert "non_negative_step" in failed
    assert "duplicate_rate" in failed
    assert "completeness_type" in failed


def test_expect_columns_exist_reports_missing(spark) -> None:
    df = _clean_frame(spark).drop("type")

    result = expect_columns_exist(df, ["step", "type", "amount"])

    assert result.success is False
    assert result.details["missing"] == "type"


def test_expect_column_completeness_handles_missing_column(spark) -> None:
    result = expect_column_completeness(_clean_frame(spark), "not_a_column")

    assert result.success is False
    assert result.details["reason"] == "column is missing"


def test_expect_binary_label_counts_invalid_values(spark) -> None:
    result = expect_binary_label(_dirty_frame(spark))

    assert result.success is False
    assert result.details["invalid_count"] == 1


def test_expect_non_negative_counts_negative_values(spark) -> None:
    result = expect_non_negative(_dirty_frame(spark), "amount")

    assert result.success is False
    assert result.details["negative_count"] == 2


def test_expect_duplicate_rate_below_threshold(spark) -> None:
    result = expect_duplicate_rate_below(_dirty_frame(spark), ["transaction_id"], max_ratio=0.01)

    assert result.success is False
    assert result.details["duplicate_ratio"] > 0.0


def test_expectation_suite_json_export_is_parseable(spark) -> None:
    suite = run_transaction_expectation_suite(_clean_frame(spark))

    payload = json.loads(expectation_suite_to_json(suite))

    assert payload["success"] is True
    assert payload["dataset_name"] == "transactions"
    assert len(payload["results"]) == 8


def test_expectation_suite_markdown_export(spark) -> None:
    suite = run_transaction_expectation_suite(_dirty_frame(spark), dataset_name="dirty")

    markdown = expectation_suite_to_markdown(suite)

    assert "# Data quality report — dirty" in markdown
    assert "FAILED" in markdown
    assert "| binary_label_isFraud | fail |" in markdown
