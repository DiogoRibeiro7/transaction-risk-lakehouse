"""Monitoring report assembly and export."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pyspark.sql import DataFrame

from transaction_risk.monitoring.drift import (
    drift_report,
    label_rate_drift,
    score_distribution_drift,
)
from transaction_risk.monitoring.performance import (
    alert_volume_by_time_bucket,
    fraud_value_by_time_bucket,
    precision_recall_by_time_bucket,
)

logger = logging.getLogger(__name__)


def build_monitoring_report(
    reference_df: DataFrame,
    current_df: DataFrame,
    feature_columns: list[str] | None = None,
    score_column: str = "fraud_probability",
    label_column: str = "isFraud",
    alert_column: str = "is_alert",
    amount_column: str = "amount",
    time_column: str = "step",
    time_bucket_size: int = 24,
    psi_bins: int = 10,
) -> dict:
    """Build a monitoring report comparing a reference and a current window.

    Labeled sections are included only when the label column is present in both
    windows, so the same report works for unlabeled production scoring output.
    """
    shared_columns = set(reference_df.columns) & set(current_df.columns)
    labeled = label_column in shared_columns

    report: dict = {
        "labeled": labeled,
        "reference_rows": reference_df.count(),
        "current_rows": current_df.count(),
    }

    drift_columns = [
        column
        for column in (feature_columns or [amount_column])
        if column in shared_columns
    ]
    if drift_columns:
        report["feature_drift_psi"] = drift_report(
            reference_df, current_df, columns=drift_columns, bins=psi_bins
        )

    if score_column in shared_columns:
        report["score_distribution_psi"] = score_distribution_drift(
            reference_df, current_df, score_column=score_column, bins=psi_bins
        )

    if labeled:
        report["label_rate_drift"] = label_rate_drift(
            reference_df, current_df, label_column=label_column
        )

    if {time_column, alert_column} <= set(current_df.columns):
        report["alert_volume_by_bucket"] = alert_volume_by_time_bucket(
            current_df,
            time_column=time_column,
            bucket_size=time_bucket_size,
            alert_column=alert_column,
        )
        if label_column in current_df.columns:
            report["precision_recall_by_bucket"] = precision_recall_by_time_bucket(
                current_df,
                time_column=time_column,
                bucket_size=time_bucket_size,
                alert_column=alert_column,
                label_column=label_column,
            )
            if amount_column in current_df.columns:
                report["fraud_value_by_bucket"] = fraud_value_by_time_bucket(
                    current_df,
                    time_column=time_column,
                    bucket_size=time_bucket_size,
                    alert_column=alert_column,
                    label_column=label_column,
                    amount_column=amount_column,
                )

    return report


def write_monitoring_report_json(report: dict, output_path: str | Path) -> None:
    """Write a monitoring report as formatted JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote monitoring report JSON to %s", path)


def _markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
    """Render a list of dictionaries as GitHub Markdown table lines."""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_monitoring_report_markdown(report: dict, output_path: str | Path) -> None:
    """Write a monitoring report as readable GitHub Markdown."""
    labeled = report.get("labeled", False)
    lines = [
        "# Monitoring report",
        "",
        f"Monitoring mode: **{'labeled' if labeled else 'unlabeled'}**",
        "",
        f"- Reference rows: {report.get('reference_rows', 'n/a')}",
        f"- Current rows: {report.get('current_rows', 'n/a')}",
    ]

    if "feature_drift_psi" in report:
        lines += ["", "## Feature drift (PSI)", ""]
        psi_rows = [
            {"feature": feature, "psi": psi}
            for feature, psi in sorted(report["feature_drift_psi"].items())
        ]
        lines += _markdown_table(psi_rows, ["feature", "psi"])

    if "score_distribution_psi" in report:
        lines += [
            "",
            "## Score distribution drift",
            "",
            f"Score PSI: {report['score_distribution_psi']:.4f}",
        ]

    if "label_rate_drift" in report:
        drift = report["label_rate_drift"]
        lines += [
            "",
            "## Label rate drift",
            "",
            f"- Reference label rate: {drift['reference_label_rate']:.4f}",
            f"- Current label rate: {drift['current_label_rate']:.4f}",
            f"- Absolute change: {drift['absolute_change']:.4f}",
            f"- Relative change: {drift['relative_change']:.4f}",
        ]
    elif not labeled:
        lines += [
            "",
            "## Label rate drift",
            "",
            "Labels are not available; labeled monitoring sections are omitted.",
        ]

    if "alert_volume_by_bucket" in report:
        lines += ["", "## Alert volume by time bucket", ""]
        lines += _markdown_table(
            report["alert_volume_by_bucket"],
            ["time_bucket", "transactions", "alerts", "alert_rate"],
        )

    if "precision_recall_by_bucket" in report:
        lines += ["", "## Precision and recall by time bucket", ""]
        lines += _markdown_table(
            report["precision_recall_by_bucket"],
            ["time_bucket", "alerts", "frauds", "captured_frauds", "precision", "recall"],
        )

    if "fraud_value_by_bucket" in report:
        lines += ["", "## Fraud value captured by time bucket", ""]
        lines += _markdown_table(
            report["fraud_value_by_bucket"],
            ["time_bucket", "fraud_amount", "captured_fraud_amount", "fraud_value_capture_rate"],
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote monitoring report Markdown to %s", path)
