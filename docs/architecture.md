# Architecture

This document describes how data and models flow through the repository, from raw input files to alerts and monitoring reports. Every node in the diagram maps to a real folder or CLI command.

## Lakehouse and model lifecycle

```mermaid
flowchart TD
    subgraph raw["Raw inputs (data/raw/)"]
        paysim["PaySim CSV<br/>paysim_sample.csv"]
        ieee["IEEE-CIS CSVs<br/>train_transaction.csv + train_identity.csv"]
    end

    subgraph lakehouse["Lakehouse layers"]
        bronze["Bronze (data/bronze/)<br/>raw, schema-enforced tables"]
        silver["Silver (data/silver/)<br/>cleaned, deduplicated transactions"]
        gold["Gold (data/gold/features)<br/>model-ready feature table"]
    end

    subgraph modelling["Modelling"]
        train["Model training<br/>transaction-risk train"]
        registry["Model registry<br/>models/registry.jsonl"]
        model["Saved Spark ML pipeline<br/>models/fraud_risk_pipeline"]
    end

    subgraph serving["Scoring"]
        batch["Batch scoring<br/>transaction-risk score-batch"]
        stream["Streaming scoring<br/>transaction-risk score-stream"]
    end

    subgraph outputs["Alerts and reports"]
        alerts["Scored alerts<br/>data/scored/ + data/streaming/scored/"]
        monitoring["Monitoring reports<br/>transaction-risk monitor"]
        quality["Data quality reports<br/>reports/quality/"]
        metrics["Training metrics<br/>reports/metrics.json"]
    end

    paysim -->|"ingest-paysim"| bronze
    ieee -->|"ingest-ieee-cis"| bronze
    bronze --> silver
    silver -->|"quality checks"| quality
    silver -->|"build-features"| gold
    gold --> train
    train --> model
    train --> metrics
    train -->|"register version"| registry
    model --> batch
    model --> stream
    gold --> batch
    silver --> stream
    batch --> alerts
    stream --> alerts
    alerts --> monitoring
```

## Layers

- **Raw** (`data/raw/`): input CSV files. PaySim-style data can be generated locally with `make sample-data`; IEEE-CIS files are downloaded with `make download-ieee-cis`. Raw data is never committed.
- **Bronze** (`data/bronze/`): raw data written as schema-enforced Parquet (or Delta) tables, preserving the source shape for reprocessing.
- **Silver** (`data/silver/`): cleaned transactions with deterministic transaction IDs, normalized types, and basic validity filters. Optional expectation checks write data quality reports here.
- **Gold** (`data/gold/features`): the model-ready feature table with transaction, temporal, entity, graph, and optional identity feature groups.
- **Model training**: temporal split, class weighting, Spark ML pipelines, fraud-specific evaluation, threshold selection, and optional probability calibration. Metrics land in `reports/metrics.json`.
- **Model registry** (`models/registry.jsonl`): versioned metadata for each training run — model path, type, metrics, threshold, and training feature table.
- **Batch scoring**: daily-style scoring of materialized feature or transaction tables, producing scores, alert flags, and the selected threshold.
- **Streaming scoring**: a local Structured Streaming demo that scores incoming CSV files micro-batch by micro-batch with the same feature pipeline.
- **Alerts and monitoring**: scored outputs feed monitoring reports covering feature drift, score drift, label-rate drift, and bucketed alert volume, precision, recall, and fraud value capture.

## Train and score lifecycle

```mermaid
sequenceDiagram
    participant U as Operator
    participant CLI as transaction-risk CLI
    participant LH as Lakehouse (bronze/silver/gold)
    participant ML as Spark ML
    participant REG as Model registry
    participant MON as Monitoring

    U->>CLI: ingest-paysim / ingest-ieee-cis
    CLI->>LH: write bronze + silver tables
    U->>CLI: build-features
    CLI->>LH: write gold feature table
    U->>CLI: train
    CLI->>ML: temporal split, fit pipeline
    ML-->>CLI: model + metrics + threshold
    CLI->>REG: append registry entry
    U->>CLI: score-batch
    CLI->>ML: load model, score features
    ML-->>LH: scored alerts table
    U->>CLI: monitor
    CLI->>MON: drift + performance report (JSON/Markdown)
```
