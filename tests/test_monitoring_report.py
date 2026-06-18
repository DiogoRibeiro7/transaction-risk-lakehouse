from __future__ import annotations

from pathlib import Path

import pytest

from transaction_risk.monitoring.drift import label_rate_drift, score_distribution_drift
from transaction_risk.monitoring.performance import (
    alert_volume_by_time_bucket,
    fraud_value_by_time_bucket,
    precision_recall_by_time_bucket,
)
from transaction_risk.monitoring.report import (
    build_monitoring_report,
    build_monitoring_report_html,
    psi_status,
    write_monitoring_report_html,
    write_monitoring_report_json,
    write_monitoring_report_markdown,
)

pytestmark = [pytest.mark.slow, pytest.mark.spark_slow]

SCORED_COLUMNS = ["step", "amount", "fraud_probability", "is_alert", "isFraud"]


def _reference_frame(spark):
    return spark.createDataFrame(
        [
            (1, 100.0, 0.10, 0, 0),
            (2, 120.0, 0.20, 0, 0),
            (3, 130.0, 0.30, 0, 0),
            (4, 50_000.0, 0.90, 1, 1),
        ],
        SCORED_COLUMNS,
    )


def _current_frame(spark):
    return spark.createDataFrame(
        [
            (25, 110.0, 0.15, 0, 0),
            (26, 80_000.0, 0.95, 1, 1),
            (27, 90_000.0, 0.92, 1, 0),
            (49, 95_000.0, 0.97, 1, 1),
            (50, 140.0, 0.25, 0, 1),
        ],
        SCORED_COLUMNS,
    )


def test_score_distribution_drift_returns_non_negative_psi(spark) -> None:
    psi = score_distribution_drift(_reference_frame(spark), _current_frame(spark))
    assert psi >= 0.0


def test_label_rate_drift_compares_windows(spark) -> None:
    drift = label_rate_drift(_reference_frame(spark), _current_frame(spark))

    assert drift["reference_label_rate"] == pytest.approx(0.25)
    assert drift["current_label_rate"] == pytest.approx(0.6)
    assert drift["absolute_change"] == pytest.approx(0.35)
    assert drift["relative_change"] == pytest.approx(1.4)


def test_alert_volume_by_time_bucket(spark) -> None:
    buckets = alert_volume_by_time_bucket(_current_frame(spark), bucket_size=24)

    by_bucket = {row["time_bucket"]: row for row in buckets}
    assert set(by_bucket) == {1, 2}
    assert by_bucket[1]["transactions"] == 3
    assert by_bucket[1]["alerts"] == 2
    assert by_bucket[2]["transactions"] == 2
    assert by_bucket[2]["alerts"] == 1


def test_precision_recall_by_time_bucket(spark) -> None:
    buckets = precision_recall_by_time_bucket(_current_frame(spark), bucket_size=24)

    by_bucket = {row["time_bucket"]: row for row in buckets}
    assert by_bucket[1]["precision"] == 0.5
    assert by_bucket[1]["recall"] == 1.0
    assert by_bucket[2]["precision"] == 1.0
    assert by_bucket[2]["recall"] == 0.5


def test_fraud_value_by_time_bucket(spark) -> None:
    buckets = fraud_value_by_time_bucket(_current_frame(spark), bucket_size=24)

    by_bucket = {row["time_bucket"]: row for row in buckets}
    assert by_bucket[1]["fraud_amount"] == 80_000.0
    assert by_bucket[1]["captured_fraud_amount"] == 80_000.0
    assert by_bucket[2]["fraud_amount"] == 95_140.0
    assert by_bucket[2]["captured_fraud_amount"] == 95_000.0


def test_build_monitoring_report_labeled(spark) -> None:
    report = build_monitoring_report(_reference_frame(spark), _current_frame(spark))

    assert report["labeled"] is True
    assert report["reference_rows"] == 4
    assert report["current_rows"] == 5
    assert "amount" in report["feature_drift_psi"]
    assert "score_distribution_psi" in report
    assert "label_rate_drift" in report
    assert "alert_volume_by_bucket" in report
    assert "precision_recall_by_bucket" in report
    assert "fraud_value_by_bucket" in report


def test_build_monitoring_report_unlabeled(spark) -> None:
    reference = _reference_frame(spark).drop("isFraud")
    current = _current_frame(spark).drop("isFraud")

    report = build_monitoring_report(reference, current)

    assert report["labeled"] is False
    assert "label_rate_drift" not in report
    assert "precision_recall_by_bucket" not in report
    assert "fraud_value_by_bucket" not in report
    assert "alert_volume_by_bucket" in report


def test_monitoring_report_exports(spark, tmp_path: Path) -> None:
    report = build_monitoring_report(_reference_frame(spark), _current_frame(spark))
    json_path = tmp_path / "monitoring.json"
    md_path = tmp_path / "monitoring.md"

    write_monitoring_report_json(report, json_path)
    write_monitoring_report_markdown(report, md_path)

    assert json_path.exists()
    markdown = md_path.read_text(encoding="utf-8")
    assert "# Monitoring report" in markdown
    assert "labeled" in markdown
    assert "## Feature drift (PSI)" in markdown
    assert "| time_bucket |" in markdown


def test_monitoring_report_markdown_marks_unlabeled(spark, tmp_path: Path) -> None:
    reference = _reference_frame(spark).drop("isFraud")
    current = _current_frame(spark).drop("isFraud")
    report = build_monitoring_report(reference, current)
    md_path = tmp_path / "monitoring_unlabeled.md"

    write_monitoring_report_markdown(report, md_path)

    markdown = md_path.read_text(encoding="utf-8")
    assert "**unlabeled**" in markdown
    assert "Labels are not available" in markdown


# --- HTML export (no Spark required) ---


def test_psi_status_thresholds() -> None:
    assert psi_status(0.05) == ("stable", "ok")
    assert psi_status(0.10) == ("moderate drift", "warn")
    assert psi_status(0.24) == ("moderate drift", "warn")
    assert psi_status(0.25) == ("significant drift", "alert")
    assert psi_status(0.9) == ("significant drift", "alert")


def test_build_html_renders_cards_tables_and_status() -> None:
    report = {
        "labeled": True,
        "reference_rows": 4,
        "current_rows": 5,
        "feature_drift_psi": {"amount": 0.30},
        "score_distribution_psi": 0.05,
        "label_rate_drift": {
            "reference_label_rate": 0.25,
            "current_label_rate": 0.60,
            "absolute_change": 0.35,
            "relative_change": 1.4,
        },
        "alert_volume_by_bucket": [
            {"time_bucket": 0, "transactions": 3, "alerts": 2, "alert_rate": 0.6667}
        ],
    }

    html_text = build_monitoring_report_html(report)

    assert html_text.startswith("<!DOCTYPE html>")
    assert "Monitoring report" in html_text
    assert "labeled" in html_text
    # PSI 0.30 is significant -> alert card; score PSI 0.05 -> ok card
    assert "card-alert" in html_text
    assert "card-ok" in html_text
    assert "significant drift" in html_text
    assert "Alert volume by time bucket" in html_text
    assert "<table" in html_text


def test_build_html_unlabeled_notes_missing_labels() -> None:
    report = {
        "labeled": False,
        "reference_rows": 4,
        "current_rows": 5,
        "feature_drift_psi": {"amount": 0.02},
        "alert_volume_by_bucket": [
            {"time_bucket": 0, "transactions": 3, "alerts": 2, "alert_rate": 0.6667}
        ],
    }

    html_text = build_monitoring_report_html(report)

    assert "unlabeled" in html_text
    assert "Labels are not available" in html_text
    assert "Label rate change" not in html_text


def test_build_html_escapes_dynamic_text() -> None:
    report = {
        "labeled": False,
        "reference_rows": 1,
        "current_rows": 1,
        "feature_drift_psi": {"amount<script>": 0.4},
    }

    html_text = build_monitoring_report_html(report)

    assert "<script>" not in html_text
    assert "&lt;script&gt;" in html_text


def test_write_html_creates_file(spark, tmp_path: Path) -> None:
    report = build_monitoring_report(_reference_frame(spark), _current_frame(spark))
    html_path = tmp_path / "monitoring.html"

    write_monitoring_report_html(report, html_path)

    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert content.startswith("<!DOCTYPE html>")
    assert "</html>" in content
