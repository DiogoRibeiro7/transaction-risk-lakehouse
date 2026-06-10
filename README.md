# transaction-risk-lakehouse

A production-oriented PySpark project for transaction risk modelling and fraud detection.

This repository demonstrates how to build a scalable transaction-risk pipeline using batch processing, behavioural feature engineering, temporal validation, graph-derived risk signals, Spark MLlib models, and a local streaming scoring demo.

The project is designed as a portfolio repository. It is not only a modelling notebook. It has package code, tests, configuration files, CLI commands, and a clear lakehouse-style data layout.

## What this project demonstrates

- PySpark DataFrame pipelines
- Spark SQL feature engineering
- Bronze / silver / gold lakehouse layout
- Fraud-specific behavioural features
- Temporal train / validation / test splits
- Imbalanced classification metrics
- Spark MLlib modelling
- Graph-derived transaction features without requiring external graph libraries
- Local streaming scoring demo with Spark Structured Streaming
- Model monitoring and drift reports
- Dockerized local execution
- Tests and CI

## Dataset

The default dataset target is **PaySim**, a synthetic mobile-money transaction dataset that is large enough to justify Spark. The raw CSV is not included in this repository.

Expected raw file:

```text
./data/raw/paysim.csv
```

Expected columns:

```text
step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
```

You can still run the project without the full dataset by generating a small local sample:

```bash
make sample-data
```

The repository also supports **IEEE-CIS Fraud Detection** ingestion when you have the Kaggle source files locally. Expected file names:

```text
./data/raw/train_transaction.csv
./data/raw/train_identity.csv
```

To fetch those files locally with the official Kaggle CLI, first authenticate with Kaggle and accept the IEEE-CIS competition rules in your browser. Kaggle’s current CLI docs describe installation with `pip install kaggle`, interactive auth via `kaggle auth login`, and legacy token-file auth with `~/.kaggle/kaggle.json`. After that, this repo provides:

```bash
make download-ieee-cis
```

That target runs:

```bash
poetry run python scripts/download_kaggle_ieee_cis.py --output-dir data/raw
```

The helper downloads the competition archive with `kaggle competitions download -c ieee-fraud-detection`, then extracts only:

- `train_transaction.csv`
- `train_identity.csv`

## Architecture

```text
raw CSV
  -> bronze Parquet    # schema-enforced raw transaction table
  -> silver Parquet    # cleaned, typed, deduplicated transactions
  -> gold Parquet      # model-ready features
  -> model artifacts   # Spark ML pipeline model
  -> alerts            # scored transactions above threshold
```

## Repository structure

```text
transaction-risk-lakehouse/
├── conf/                    # pipeline configuration
├── data/                    # local data folders, ignored by git except .gitkeep
├── notebooks/               # portfolio notebooks
├── reports/                 # generated figures and model card
├── scripts/                 # helper scripts
├── src/transaction_risk/     # production package code
├── tests/                   # unit tests
├── docker-compose.yml       # local Spark environment
├── pyproject.toml           # Poetry project
├── ROADMAP.md
└── AGENTS.md
```

## Quick start

### 1. Install dependencies

```bash
poetry install
```

PySpark requires Java. Use Java 17 locally unless your Spark distribution requires otherwise.

### 2. Generate sample data

```bash
make sample-data
```

If you want the real IEEE-CIS training files instead of the synthetic PaySim sample:

```bash
make download-ieee-cis
```

### 3. Build lakehouse tables

```bash
make ingest
make features
```

### 4. Train a model

```bash
make train
```

### 5. Run tests

```bash
make test
```

## CLI examples

```bash
transaction-risk ingest-paysim \
  --input data/raw/paysim_sample.csv \
  --bronze data/bronze/paysim \
  --silver data/silver/transactions \
  --table-format parquet
```

```bash
transaction-risk build-features \
  --input data/silver/transactions \
  --output data/gold/features \
  --table-format parquet
```

```bash
transaction-risk train \
  --input data/gold/features \
  --model-output models/fraud_risk_pipeline \
  --metrics-output reports/metrics.json \
  --table-format parquet
```

```bash
transaction-risk ingest-ieee-cis \
  --transaction-input data/raw/train_transaction.csv \
  --identity-input data/raw/train_identity.csv \
  --bronze data/bronze/ieee_cis \
  --silver data/silver/ieee_cis_transactions
```

## Optional Delta Lake support

The base project uses Parquet by default and does not require Delta Lake dependencies.

If your Spark environment is configured with Delta Lake support, the ingestion, feature, and training commands also accept:

```bash
--table-format delta
```

If Delta is requested in an environment without Delta support, the CLI raises a clear error and the default Parquet path remains unchanged.

## Modelling strategy

The project avoids random train/test leakage. Transactions are split by time using `step`, which represents ordered transaction time in PaySim. This gives a more realistic validation scheme for fraud detection than a random split.

The default model pipeline includes:

- categorical encoding for transaction type
- vector assembly
- feature scaling where appropriate
- logistic regression baseline
- random forest candidate model
- gradient-boosted tree candidate model

The evaluation module reports:

- ROC-AUC
- PR-AUC
- precision at top K
- recall at top K
- alert volume
- confusion matrix at threshold
- fraud value captured

Operational threshold selection supports both capacity-based and business-value-based strategies. The default remains alert-rate targeting, and training can also optimize for expected value:

```bash
transaction-risk train \
  --input data/gold/features \
  --model-output models/fraud_risk_pipeline \
  --metrics-output reports/metrics.json \
  --threshold-strategy expected-value \
  --fraud-loss-rate 1.0 \
  --false-positive-review-cost 5.0 \
  --true-positive-recovery-rate 0.6
```

## Feature groups

### Transaction features

- log amount
- balance deltas
- balance inconsistency flags
- destination merchant indicator
- zero-balance flags
- transaction type one-hot features through Spark ML

### Temporal features

- previous transaction step per origin account
- time since previous transaction
- rolling transaction count by account
- rolling amount statistics by account

### Entity features

- origin account historical transaction count
- destination historical transaction count
- origin average amount
- destination average received amount
- origin-to-destination pair frequency

### Graph features

The graph module computes Spark-native graph features using DataFrame aggregations:

- origin out-degree
- destination in-degree
- repeated counterparties
- pair frequency
- neighbourhood fraud exposure, if labels are available

This avoids forcing GraphFrames for the base project while still showing graph-aware fraud engineering.

### IEEE-CIS identity features

When `enable_identity_features: true` is set in `conf/features.yaml`, the feature pipeline can also add optional IEEE-CIS-specific features such as:

- card-field presence indicators
- normalized purchaser and recipient email domains
- email-domain match flags
- normalized device-type fields
- identity missingness counts
- product-code transaction amount aggregates

The identity feature group is resilient to partially populated IEEE-CIS inputs and does not change the default PaySim path.

## Feature registry

The repository includes a lightweight feature registry for generated model features. It records ownership, source columns, defaults, and leakage notes without requiring Spark at export time.

```bash
transaction-risk export-feature-registry --output reports/feature_registry.md
```

## Streaming demo

The streaming demo reads incoming CSV files from a local folder and writes scored alerts to Parquet.

```bash
transaction-risk score-stream \
  --input-stream data/streaming/incoming \
  --model models/fraud_risk_pipeline \
  --output data/streaming/scored \
  --checkpoint data/streaming/checkpoints/scoring
```

## Development

```bash
make format
make lint
make test
```

## Portfolio narrative

This repository is designed to support the following story:

> I built a scalable PySpark transaction-risk pipeline that transforms raw transaction logs into behavioural, temporal, and graph-derived fraud-risk features, trains Spark-native models, evaluates them with temporal validation, and simulates production alerting.

## License

MIT.
