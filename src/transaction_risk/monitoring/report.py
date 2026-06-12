"""Monitoring report assembly and export."""

from __future__ import annotations

import html
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


# Standard population stability index interpretation thresholds.
PSI_MODERATE_THRESHOLD = 0.1
PSI_SIGNIFICANT_THRESHOLD = 0.25


def psi_status(psi: float) -> tuple[str, str]:
    """Map a PSI value to an Evidently-style status label and severity class."""
    if psi >= PSI_SIGNIFICANT_THRESHOLD:
        return "significant drift", "alert"
    if psi >= PSI_MODERATE_THRESHOLD:
        return "moderate drift", "warn"
    return "stable", "ok"


def _format_cell(value: object) -> str:
    """Format a single table cell, escaping any dynamic text."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))


def _html_table(rows: list[dict], columns: list[str]) -> str:
    """Render a list of dictionaries as an HTML table."""
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{_format_cell(row.get(column, ''))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        '<table class="report-table">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _metric_card(title: str, value: str, status_label: str, severity: str) -> str:
    """Render a single status card."""
    return (
        f'<div class="card card-{severity}">'
        f'<div class="card-title">{html.escape(title)}</div>'
        f'<div class="card-value">{html.escape(value)}</div>'
        f'<div class="card-status">{html.escape(status_label)}</div>'
        "</div>"
    )


_HTML_STYLE = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 24px; background: #f4f6f8; color: #1f2933; }
h1 { margin: 0 0 4px; font-size: 24px; }
h2 { margin: 28px 0 12px; font-size: 18px; }
.subtitle { color: #52606d; margin-bottom: 20px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px;
         font-weight: 600; background: #d9e2ec; color: #243b53; }
.cards { display: flex; flex-wrap: wrap; gap: 16px; margin: 8px 0 4px; }
.card { flex: 1 1 200px; background: #fff; border-radius: 10px; padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-left: 4px solid #9aa5b1; }
.card-ok { border-left-color: #27ab83; }
.card-warn { border-left-color: #f0b429; }
.card-alert { border-left-color: #cf1124; }
.card-title { font-size: 13px; color: #52606d; }
.card-value { font-size: 22px; font-weight: 700; margin: 6px 0; }
.card-status { font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: #616e7c; }
.report-table { border-collapse: collapse; width: 100%; background: #fff; border-radius: 8px;
                overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.report-table th { background: #f0f4f8; text-align: left; padding: 8px 12px; font-size: 13px; }
.report-table td { padding: 8px 12px; border-top: 1px solid #e4e7eb; font-size: 13px; }
.note { color: #616e7c; font-style: italic; }
"""


def build_monitoring_report_html(report: dict, title: str = "Monitoring report") -> str:
    """Render a monitoring report as a self-contained Evidently-style HTML dashboard.

    The output is a single HTML document with inline CSS and no external assets or
    dependencies, so it can be opened directly in a browser or attached to a review.
    """
    labeled = report.get("labeled", False)
    mode = "labeled" if labeled else "unlabeled"

    cards: list[str] = []
    if "feature_drift_psi" in report and report["feature_drift_psi"]:
        worst_feature, worst_psi = max(report["feature_drift_psi"].items(), key=lambda item: item[1])
        status_label, severity = psi_status(float(worst_psi))
        cards.append(
            _metric_card(
                f"Top feature drift ({html.escape(worst_feature)})",
                f"{float(worst_psi):.4f}",
                status_label,
                severity,
            )
        )
    if "score_distribution_psi" in report:
        score_psi = float(report["score_distribution_psi"])
        status_label, severity = psi_status(score_psi)
        cards.append(_metric_card("Score distribution PSI", f"{score_psi:.4f}", status_label, severity))
    if "label_rate_drift" in report:
        drift = report["label_rate_drift"]
        relative = float(drift["relative_change"])
        severity = "alert" if abs(relative) >= 0.5 else "warn" if abs(relative) >= 0.1 else "ok"
        cards.append(
            _metric_card(
                "Label rate change",
                f"{float(drift['reference_label_rate']):.4f} -> {float(drift['current_label_rate']):.4f}",
                f"{relative:+.1%} relative",
                severity,
            )
        )

    sections: list[str] = []
    if "feature_drift_psi" in report:
        psi_rows = [
            {"feature": feature, "psi": psi, "status": psi_status(float(psi))[0]}
            for feature, psi in sorted(report["feature_drift_psi"].items())
        ]
        sections.append("<h2>Feature drift (PSI)</h2>")
        sections.append(_html_table(psi_rows, ["feature", "psi", "status"]))

    if "label_rate_drift" not in report and not labeled:
        sections.append("<h2>Labeled monitoring</h2>")
        sections.append(
            '<p class="note">Labels are not available; labeled sections '
            "(label-rate drift, precision/recall, fraud value) are omitted.</p>"
        )

    if "alert_volume_by_bucket" in report:
        sections.append("<h2>Alert volume by time bucket</h2>")
        sections.append(
            _html_table(
                report["alert_volume_by_bucket"],
                ["time_bucket", "transactions", "alerts", "alert_rate"],
            )
        )
    if "precision_recall_by_bucket" in report:
        sections.append("<h2>Precision and recall by time bucket</h2>")
        sections.append(
            _html_table(
                report["precision_recall_by_bucket"],
                ["time_bucket", "alerts", "frauds", "captured_frauds", "precision", "recall"],
            )
        )
    if "fraud_value_by_bucket" in report:
        sections.append("<h2>Fraud value captured by time bucket</h2>")
        sections.append(
            _html_table(
                report["fraud_value_by_bucket"],
                ["time_bucket", "fraud_amount", "captured_fraud_amount", "fraud_value_capture_rate"],
            )
        )

    cards_html = f'<div class="cards">{"".join(cards)}</div>' if cards else ""
    return (
        "<!DOCTYPE html>"
        '<html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        f"<style>{_HTML_STYLE}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f'<div class="subtitle">Monitoring mode: <span class="badge">{html.escape(mode)}</span>'
        f" &middot; reference rows: {html.escape(str(report.get('reference_rows', 'n/a')))}"
        f" &middot; current rows: {html.escape(str(report.get('current_rows', 'n/a')))}</div>"
        f"{cards_html}"
        f"{''.join(sections)}"
        "</body></html>"
    )


def write_monitoring_report_html(report: dict, output_path: str | Path) -> None:
    """Write a monitoring report as a self-contained HTML dashboard."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_monitoring_report_html(report), encoding="utf-8")
    logger.info("Wrote monitoring report HTML to %s", path)


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
