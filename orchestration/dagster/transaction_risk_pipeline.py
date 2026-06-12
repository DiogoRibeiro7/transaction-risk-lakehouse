"""Dagster orchestration example for the transaction risk pipeline.

The pipeline does not duplicate business logic. Every step shells out to the
project CLI (or a project script), so the orchestrated path stays identical to
the documented manual workflow. Dagster is an optional dependency: this module
still imports without it so the command definitions remain testable.
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

PIPELINE_STEPS: dict[str, list[str]] = {
    "generate_sample_data": [
        "poetry",
        "run",
        "python",
        "scripts/generate_sample_paysim.py",
        "--output",
        "data/raw/paysim_sample.csv",
        "--rows",
        "5000",
        "--fraud-rate",
        "0.015",
    ],
    "ingest_raw_data": [
        "poetry",
        "run",
        "transaction-risk",
        "ingest-paysim",
        "--input",
        "data/raw/paysim_sample.csv",
        "--bronze",
        "data/bronze/paysim",
        "--silver",
        "data/silver/transactions",
        "--write-quality-report",
        "reports/quality/paysim_ingestion.json",
    ],
    "build_features": [
        "poetry",
        "run",
        "transaction-risk",
        "build-features",
        "--input",
        "data/silver/transactions",
        "--output",
        "data/gold/features",
    ],
    "train_model": [
        "poetry",
        "run",
        "transaction-risk",
        "train",
        "--input",
        "data/gold/features",
        "--model-output",
        "models/fraud_risk_pipeline",
        "--metrics-output",
        "reports/metrics.json",
    ],
    "score_batch": [
        "poetry",
        "run",
        "transaction-risk",
        "score-batch",
        "--input",
        "data/gold/features",
        "--model",
        "models/fraud_risk_pipeline",
        "--output",
        "data/scored/batch",
        "--threshold",
        "0.5",
    ],
    "monitoring_report": [
        "poetry",
        "run",
        "transaction-risk",
        "monitor",
        "--reference",
        "data/scored/batch",
        "--current",
        "data/scored/batch",
        "--output-json",
        "reports/monitoring/report.json",
        "--output-md",
        "reports/monitoring/report.md",
    ],
}


def build_step_command(step_name: str) -> list[str]:
    """Return the CLI command for a pipeline step."""
    if step_name not in PIPELINE_STEPS:
        known_steps = ", ".join(sorted(PIPELINE_STEPS))
        raise ValueError(f"Unknown pipeline step '{step_name}'. Known steps: {known_steps}.")
    return list(PIPELINE_STEPS[step_name])


def run_step(step_name: str) -> None:
    """Run a pipeline step as a subprocess and fail loudly on errors."""
    command = build_step_command(step_name)
    logger.info("Running pipeline step %s: %s", step_name, " ".join(command))
    subprocess.run(command, check=True)


try:  # pragma: no cover - exercised only when Dagster is installed
    from dagster import In, Nothing, job, op

    HAS_DAGSTER = True
except ImportError:  # pragma: no cover
    HAS_DAGSTER = False


if HAS_DAGSTER:

    @op
    def generate_sample_data() -> None:
        """Generate the synthetic PaySim sample CSV."""
        run_step("generate_sample_data")

    @op(ins={"start": In(Nothing)})
    def ingest_raw_data() -> None:
        """Ingest raw data into bronze and silver tables."""
        run_step("ingest_raw_data")

    @op(ins={"start": In(Nothing)})
    def build_features() -> None:
        """Build the gold feature table."""
        run_step("build_features")

    @op(ins={"start": In(Nothing)})
    def train_model() -> None:
        """Train and register the fraud model."""
        run_step("train_model")

    @op(ins={"start": In(Nothing)})
    def score_batch() -> None:
        """Score the feature table with the trained model."""
        run_step("score_batch")

    @op(ins={"start": In(Nothing)})
    def monitoring_report() -> None:
        """Generate monitoring reports from scored output."""
        run_step("monitoring_report")

    @job
    def transaction_risk_job() -> None:
        """End-to-end sample pipeline: data, ingestion, features, training, scoring, monitoring."""
        sample = generate_sample_data()
        silver = ingest_raw_data(start=sample)
        features = build_features(start=silver)
        model = train_model(start=features)
        scored = score_batch(start=model)
        monitoring_report(start=scored)
