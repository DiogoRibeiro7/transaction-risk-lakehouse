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
  --silver data/silver/transactions
```

```bash
transaction-risk build-features \
  --input data/silver/transactions \
  --output data/gold/features
```

```bash
transaction-risk train \
  --input data/gold/features \
  --model-output models/fraud_risk_pipeline \
  --metrics-output reports/metrics.json
```

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
