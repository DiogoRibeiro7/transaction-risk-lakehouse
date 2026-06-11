# Orchestration example

This folder contains an optional Dagster example that orchestrates the full sample pipeline:

1. generate sample data
2. ingest raw data (with a data quality report)
3. build features
4. train the model
5. score a batch
6. generate a monitoring report

Each step shells out to the project CLI, so the orchestrated pipeline runs exactly the same code as the documented manual workflow. The monitoring step compares the scored batch with itself purely as a wiring demo; in production the reference and current inputs would be different time windows.

## Setup

Dagster is not a base dependency. Install it with the optional group:

```bash
poetry install --with orchestration
```

## Run the pipeline locally

Launch the Dagster UI and run the `transaction_risk_job`:

```bash
poetry run dagster dev -f orchestration/dagster/transaction_risk_pipeline.py
```

Or execute the job directly without the UI:

```bash
poetry run dagster job execute -f orchestration/dagster/transaction_risk_pipeline.py -j transaction_risk_job
```

## Notes

- The module imports without Dagster installed, so the step definitions stay testable in the base environment.
- Commands are defined in `PIPELINE_STEPS` inside `orchestration/dagster/transaction_risk_pipeline.py`.
