# Project walkthrough

This is a narrative walkthrough of `transaction-risk-lakehouse` for reviewers who want to understand the project without running it. It explains what the project does, why each design decision was made, and where the honest limitations are.

## Problem framing

Fraud detection in payment systems is an imbalanced ranking and alerting problem, not just binary classification. Fraud is rare (well under 1% of transactions), labels arrive late or incompletely, and the business constraint is review capacity: an operations team can only investigate a fixed number of alerts per day. The model's job is therefore to rank transactions by risk so that the limited review budget captures as much fraud value as possible.

This repository builds that framing end to end: lakehouse ingestion, distributed feature engineering, temporal validation, Spark ML modelling, fraud-specific evaluation (precision@K, fraud value capture), operational thresholding, and monitoring.

## Dataset and assumptions

Two datasets are supported:

- **PaySim** (default): a synthetic mobile-money simulation with ordered time steps, transaction types, and account balances. A small synthetic sample can be generated locally (`make sample-data`), so the full pipeline runs without downloading anything.
- **IEEE-CIS Fraud Detection**: real anonymized e-commerce transactions with an optional identity table joined by `TransactionID` (`make download-ieee-cis` with Kaggle credentials).

Key assumptions: the `step` (PaySim) or `TransactionDT` (IEEE-CIS) column is a reliable event-time proxy; labels are trustworthy at training time; and amounts are in a single currency unit.

## Lakehouse architecture

Data flows through bronze (raw, schema-enforced), silver (cleaned, deduplicated, with deterministic transaction IDs), and gold (model-ready features) layers, written as Parquet by default with optional Delta support. See [architecture.md](architecture.md) for the diagram. The layering means any stage can be rebuilt from the previous one, and data quality checks run at the silver boundary.

## Feature engineering

Features are built with PySpark DataFrame APIs only — no pandas conversion — so the same code scales beyond a laptop:

- **Transaction features**: log amounts, balance deltas and inconsistencies, type indicators, merchant destination flags.
- **Temporal features**: per-account transaction history (time since previous transaction, rolling mean/std of amounts, z-scores) computed with Spark windows that only look backwards in time.
- **Entity features**: per-account and per-counterparty aggregates computed from prior transactions only.
- **Graph features**: in/out degree, edge frequency, and destination historical fraud rate computed from prior transactions only (no graph library needed).
- **Identity features** (IEEE-CIS): card presence, email-domain indicators, device type, and missingness counts, all robust to absent columns.

Every feature has a registry entry with owner, source columns, and leakage notes (`transaction-risk export-feature-registry`).

## Temporal validation

Random splits leak future information into training: an account's later transactions inform its earlier ones through aggregate features, and fraud patterns are non-stationary. The project always splits by time — train on the earliest window, validate on the next, test on the most recent — and historical features are built with windows that exclude the current row.

## Modelling results

Three Spark ML models are compared under the same temporal split: logistic regression (weighted, baseline), random forest, and gradient-boosted trees. `make benchmark` regenerates the comparison table (ROC-AUC, PR-AUC, precision@K, recall@K, alert count, selected threshold) in `reports/benchmark/`. Class imbalance is handled with inverse-frequency class weights rather than undersampling, so the probability scale stays interpretable.

Optional probability calibration (Platt scaling or isotonic regression, fitted on the validation window) produces calibrated probabilities with Brier score and binned expected-calibration-error reporting. When enabled, the calibrator is saved with the model artifact and used by batch and streaming scoring, so deployed probabilities match the reported calibrated metrics.

## Operational thresholding

Two threshold strategies are supported:

- **Alert-rate targeting** (default): pick the score threshold that produces a target alert volume, matching review capacity.
- **Expected value**: pick the threshold that maximizes recovered fraud value minus missed fraud losses and review costs, with configurable cost parameters.

Each training run registers the model with its threshold and metrics in a local registry (`models/registry.jsonl`). The saved model path is a bundled scoring artifact directory containing the base Spark model and, when enabled, the fitted calibrator.

## Local Spark environment

The repository now makes local Spark sessions more predictable on Windows and mixed-Python environments. Session startup loads a local `.env` automatically, applies `SPARK_JAVA_HOME` and `HADOOP_HOME` when present, and pins both `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` to the active interpreter. That avoids the common failure mode where Spark workers start via the Microsoft Store Python alias instead of the real interpreter.

## Monitoring

The `transaction-risk monitor` command compares a reference scored window with a current one and writes JSON/Markdown reports covering feature PSI, score distribution drift, label-rate drift, and bucketed alert volume, precision, recall, and fraud value capture. Reports explicitly distinguish labeled from unlabeled monitoring, since production labels usually lag.

## Limitations and future work

- **Synthetic data**: PaySim does not reproduce real fraud adversarial behaviour; metrics on it demonstrate the pipeline, not production performance.
- **Label leakage risk**: destination historical fraud rate uses labels; in production it must be computed from confirmed-fraud tables with appropriate delay.
- **Label delay**: the pipeline assumes labels are available at training time; real systems need delayed-label evaluation.
- **Threshold assumptions**: expected-value parameters (loss rates, review costs) are configurable inputs, not learned quantities.
- **Single-machine demos**: streaming and orchestration examples run locally; production deployments need cluster configuration, secrets management, and CI/CD.

See [limitations.md](limitations.md) for the full discussion. The [ROADMAP](../ROADMAP.md) tracks remaining portfolio polish.
