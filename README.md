# transaction-risk-lakehouse

A production-oriented PySpark project for transaction risk modelling and fraud detection.

This repository demonstrates how to build a scalable transaction-risk pipeline using batch processing, behavioural feature engineering, temporal validation, graph-derived risk signals, Spark MLlib models, and a local streaming scoring demo.

The project is designed as a portfolio repository. It is not only a modelling notebook. It has package code, tests, configuration files, CLI commands, and a clear lakehouse-style data layout.

See [docs/architecture.md](docs/architecture.md) for the architecture diagram covering the lakehouse layers, model training and registry, batch and streaming scoring, and monitoring. For a narrative tour of the design decisions — problem framing, features, validation, thresholding, monitoring, and honest limitations — read [docs/walkthrough.md](docs/walkthrough.md). The [notebooks/](notebooks/) folder contains short runnable notebooks for EDA, feature engineering, temporal validation, model training, graph features, streaming, and monitoring; [docs/limitations.md](docs/limitations.md) covers what this project deliberately does not claim.

## What this project demonstrates

- PySpark DataFrame pipelines
- Spark SQL feature engineering
- Bronze / silver / gold lakehouse layout (Parquet by default, optional Delta)
- Fraud-specific behavioural features with a documented feature registry
- Temporal train / validation / test splits
- Imbalanced classification metrics and a reproducible model benchmark
- Spark MLlib modelling with optional probability calibration
- Cost-sensitive and capacity-based alert thresholding
- Graph-derived transaction features without requiring external graph libraries
- Batch scoring and a local streaming scoring demo with Spark Structured Streaming
- Local model registry with versioned metadata
- Lightweight data expectation checks and quality reports
- Model monitoring and drift reports (JSON and Markdown)
- Optional Dagster orchestration example
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
  -> model artifacts   # bundled scoring artifact (base Spark model + optional calibrator)
  -> alerts            # scored transactions above threshold
```

## Repository structure

```text
transaction-risk-lakehouse/
├── conf/                    # pipeline configuration
├── data/                    # local data folders, ignored by git except .gitkeep
├── docs/                    # architecture, walkthrough, and limitations
├── notebooks/               # portfolio notebooks
├── orchestration/           # optional Dagster example pipeline
├── reports/                 # generated reports and model card
├── scripts/                 # helper scripts
├── src/transaction_risk/     # production package code
├── tests/                   # unit tests
├── docker-compose.yml       # local Spark environment
├── pyproject.toml           # Poetry project
└── ROADMAP.md
```

## Quick start

### 1. Install dependencies

```bash
poetry install
```

PySpark requires Java 17 or 21. Java 24+ does not work with Spark 4 because it removes the Security Manager that Hadoop still depends on. If your machine-wide `JAVA_HOME` points to a newer JDK, set `SPARK_JAVA_HOME` to a compatible JDK in a local `.env` file. Spark sessions created through the package now load `.env` automatically before launching the JVM.

On Windows, Hadoop filesystem operations (including Spark ML model saves) additionally need `winutils.exe` and `hadoop.dll`; point `HADOOP_HOME` at a folder whose `bin` contains them (builds for each Hadoop version are available from the `cdarlint/winutils` GitHub repository). The session helper also pins `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` to the active interpreter so local workers do not fall back to the Microsoft Store Python alias.

```text
SPARK_JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-21.0.11.10-hotspot
HADOOP_HOME=C:\Users\<you>\hadoop
```

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

Historical aggregate features are built causally: account, counterparty, graph, and optional IEEE-CIS product-code aggregates only use rows strictly earlier than the current transaction.

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

### Probability calibration

Calibration is optional and off by default. When enabled, a calibrator is fitted on validation predictions (Platt scaling by default, isotonic regression optionally), used for threshold selection and test evaluation, and saved inside the deployed scoring artifact. The metrics report includes Brier score before and after calibration, a binned expected calibration error approximation, and a calibration table.

```bash
transaction-risk train \
  --input data/gold/features \
  --model-output models/fraud_risk_pipeline \
  --metrics-output reports/metrics.json \
  --calibrate-probabilities \
  --calibration-method platt
```

### Model benchmark

The benchmark command trains logistic regression, random forest, and gradient-boosted trees under the same temporal split and reports ROC-AUC, PR-AUC, precision@K, recall@K, alert count, and the selected threshold. Results are written to `reports/benchmark/metrics.json` and `reports/benchmark/metrics.md`:

```bash
make benchmark
```

| model_type | roc_auc | pr_auc | precision_at_k | recall_at_k | alert_count | selected_threshold |
| --- | --- | --- | --- | --- | --- | --- |
| _generated_ | — | — | — | — | — | — |

Run `make benchmark` on your data to populate this table from `reports/benchmark/metrics.md`.

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

## Model registry

Each successful training run appends a versioned entry to a local JSON Lines registry under `models/registry.jsonl`. An entry records the model path, model type, training feature table, selected threshold, evaluation metrics, and registration time. The registry functions are plain Python and do not require Spark.

`--model-output` now writes a bundled scoring artifact directory. By default it contains:

```text
models/fraud_risk_pipeline/
  artifact_metadata.json
  base_model/
  calibrator/           # only when --calibrate-probabilities is enabled
```

```bash
transaction-risk list-models --registry models/registry.jsonl
```

## Batch scoring

Batch scoring supports production-style daily scoring of already materialized tables. It can score gold feature tables directly, or score cleaned silver transactions by building features first. The output contains the deployed fraud score, alert flag, selected threshold, and the input identifier columns. When the saved artifact includes a calibrator, scoring applies it automatically and preserves the uncalibrated score as `uncalibrated_fraud_probability`.

```bash
transaction-risk score-batch \
  --input data/gold/features \
  --model models/fraud_risk_pipeline \
  --output data/scored/batch \
  --threshold 0.5 \
  --input-kind features
```

```bash
transaction-risk score-batch \
  --input data/silver/transactions \
  --model models/fraud_risk_pipeline \
  --output data/scored/batch \
  --threshold 0.5 \
  --input-kind transactions
```

There is also a Makefile shortcut:

```bash
make score-batch
```

## Streaming demo

The streaming demo reads incoming CSV files from a local folder and writes scored alerts to Parquet. It loads the same bundled scoring artifact used by batch scoring, so optional calibration is applied there as well.

```bash
transaction-risk score-stream \
  --input-stream data/streaming/incoming \
  --model models/fraud_risk_pipeline \
  --output data/streaming/scored \
  --checkpoint data/streaming/checkpoints/scoring
```

## Data quality checks

The repository includes a lightweight expectation layer inspired by Great Expectations, without depending on the Great Expectations runtime. The transaction suite checks required columns, column completeness, binary label validity, non-negative amounts and steps, and the duplicate rate. Checks return structured results instead of raising, and ingestion can record them with an optional flag:

```bash
transaction-risk ingest-paysim \
  --input data/raw/paysim_sample.csv \
  --bronze data/bronze/paysim \
  --silver data/silver/transactions \
  --write-quality-report reports/quality/paysim_ingestion.json
```

Add `--strict-quality` to fail ingestion when any expectation does not pass. A `.md` suffix on the report path writes Markdown instead of JSON.

## Monitoring reports

The monitoring command compares a reference scored table with a current scored table and writes JSON, Markdown, and HTML reports. It always reports feature drift, score distribution drift, and alert volume by time bucket; when labels are present it also reports label-rate drift, precision and recall by bucket, and fraud value captured by bucket. Unlabeled tables produce a clearly marked unlabeled report instead of failing.

```bash
transaction-risk monitor \
  --reference data/scored/reference \
  --current data/scored/current \
  --output-json reports/monitoring/report.json \
  --output-md reports/monitoring/report.md \
  --output-html reports/monitoring/report.html \
  --feature-columns amount,fraud_probability
```

The optional `--output-html` flag writes a self-contained, Evidently-style HTML dashboard (drift status cards with PSI severity coloring and per-bucket performance tables). It has no external assets or dependencies — open it directly in a browser or attach it to a review — and the base project does not depend on the heavy `evidently` package.

## Demo figures

`make demo-artifacts` renders small figures from the sample pipeline outputs into [reports/figures/](reports/figures/) — class imbalance, fraud probability distribution, alert volume by threshold, and (after `make benchmark`) a model metric comparison. Figures are generated locally and not committed; see [reports/figures/README.md](reports/figures/README.md). GIF assembly is optional and only runs when `imageio` is installed.

## Orchestration example

An optional Dagster pipeline under [orchestration/](orchestration/) wires the full sample workflow (sample data, ingestion, features, training, batch scoring, monitoring) by calling the project CLI, without duplicating business logic. Dagster stays out of the base dependencies:

```bash
poetry install --with orchestration
poetry run dagster dev -f orchestration/dagster/transaction_risk_pipeline.py
```

See [orchestration/README.md](orchestration/README.md) for details.

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
