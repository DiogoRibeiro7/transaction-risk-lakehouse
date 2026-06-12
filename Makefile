.PHONY: install sample-data download-ieee-cis ingest features train score-batch benchmark demo-artifacts test lint format clean

install:
	poetry install

sample-data:
	poetry run python scripts/generate_sample_paysim.py --output data/raw/paysim_sample.csv --rows 5000 --fraud-rate 0.015

download-ieee-cis:
	poetry run python scripts/download_kaggle_ieee_cis.py --output-dir data/raw

ingest:
	poetry run transaction-risk ingest-paysim --input data/raw/paysim_sample.csv --bronze data/bronze/paysim --silver data/silver/transactions

features:
	poetry run transaction-risk build-features --input data/silver/transactions --output data/gold/features

train:
	poetry run transaction-risk train --input data/gold/features --model-output models/fraud_risk_pipeline --metrics-output reports/metrics.json

score-batch:
	poetry run transaction-risk score-batch --input data/gold/features --model models/fraud_risk_pipeline --output data/scored/batch --threshold 0.5

benchmark:
	poetry run transaction-risk benchmark-models --input data/gold/features --output reports/benchmark

demo-artifacts:
	poetry run python scripts/generate_demo_artifacts.py

test:
	poetry run pytest

lint:
	poetry run ruff check src tests
	poetry run mypy src

format:
	poetry run ruff format src tests scripts
	poetry run ruff check --fix src tests scripts

clean:
	rm -rf data/bronze data/silver data/gold data/scored data/streaming/checkpoints models reports/metrics.json
