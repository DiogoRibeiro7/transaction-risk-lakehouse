from __future__ import annotations

import pytest

from transaction_risk.experiments.benchmark import (
    REQUIRED_RESULT_KEYS,
    benchmark_results_to_markdown,
    validate_benchmark_result,
    write_benchmark_reports,
)


def _result(model_type: str = "logistic_regression") -> dict:
    return {
        "model_type": model_type,
        "roc_auc": 0.93,
        "pr_auc": 0.41,
        "precision_at_k": 0.30,
        "recall_at_k": 0.62,
        "alert_count": 50,
        "selected_threshold": 0.82,
    }


def test_validate_benchmark_result_accepts_complete_result() -> None:
    validate_benchmark_result(_result())


def test_validate_benchmark_result_rejects_missing_keys() -> None:
    incomplete = _result()
    del incomplete["pr_auc"]
    del incomplete["alert_count"]

    with pytest.raises(ValueError, match="alert_count, pr_auc"):
        validate_benchmark_result(incomplete)


def test_benchmark_markdown_contains_all_models_and_columns() -> None:
    results = [_result("logistic_regression"), _result("gbt")]

    markdown = benchmark_results_to_markdown(results)

    assert "# Model benchmark" in markdown
    assert "temporal train/validation/test split" in markdown
    assert "| model_type |" in markdown
    assert "| logistic_regression |" in markdown
    assert "| gbt |" in markdown
    assert "0.9300" in markdown
    assert "| 50 |" in markdown


def test_benchmark_markdown_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        benchmark_results_to_markdown([])


def test_write_benchmark_reports_creates_json_and_markdown(tmp_path) -> None:
    import json

    results = [_result()]
    write_benchmark_reports(results, tmp_path / "benchmark")

    json_path = tmp_path / "benchmark" / "metrics.json"
    md_path = tmp_path / "benchmark" / "metrics.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload[0]["model_type"] == "logistic_regression"
    assert set(payload[0]) >= REQUIRED_RESULT_KEYS
