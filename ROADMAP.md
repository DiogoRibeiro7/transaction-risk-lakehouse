# Roadmap

## Phase 1 — Repository and Spark foundation

- [x] Create Poetry project
- [x] Add package layout under `src/transaction_risk`
- [x] Add Spark session factory
- [x] Add local configuration files
- [x] Add Docker Compose for Spark
- [x] Add Makefile commands
- [x] Add CI workflow

## Phase 2 — Lakehouse ingestion

- [x] Define PaySim schema
- [x] Read raw CSV with strict schema
- [x] Write bronze Parquet table
- [x] Clean and deduplicate silver transactions
- [x] Add data quality checks
- [ ] Add optional Delta Lake support
- [ ] Add IEEE-CIS ingestion module

## Phase 3 — Feature engineering

- [x] Transaction-level features
- [x] Balance-consistency features
- [x] Entity-level aggregation features
- [x] Temporal window features
- [x] Spark-native graph-derived features
- [ ] Add device / identity feature group for IEEE-CIS
- [ ] Add feature store metadata

## Phase 4 — Modelling

- [x] Rule-based baseline
- [x] Logistic regression pipeline
- [x] Random forest pipeline
- [x] Gradient-boosted tree pipeline
- [x] Temporal split utility
- [x] PR-AUC and ROC-AUC evaluation
- [x] Precision@K and recall@K
- [x] Threshold selection
- [ ] Add cost-sensitive threshold optimization
- [ ] Add calibrated probability output

## Phase 5 — Monitoring and operations

- [x] Population stability index helper
- [x] Alert summary report
- [x] Model card template
- [x] Local streaming scoring job
- [ ] Add Evidently-style report export
- [ ] Add Great Expectations checks
- [ ] Add Airflow or Dagster example pipeline

## Phase 6 — Portfolio polish

- [ ] Add architecture diagram
- [ ] Add notebook screenshots
- [ ] Add benchmark table in README
- [ ] Add synthetic data demo GIF
- [ ] Add blog-style walkthrough
