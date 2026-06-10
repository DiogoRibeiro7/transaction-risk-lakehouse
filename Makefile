.PHONY: install sample-data ingest features train test lint format clean

install:
	poetry install

sample-data:
	poetry run python scripts/generate_sample_paysim.py --output data/raw/paysim_sample.csv --rows 5000 --fraud-rate 0.015

ingest:
	poetry run transaction-risk ingest-paysim --input data/raw/paysim_sample.csv --bronze data/bronze/paysim --silver data/silver/transactions

features:
	poetry run transaction-risk build-features --input data/silver/transactions --output data/gold/features

train:
	poetry run transaction-risk train --input data/gold/features --model-output models/fraud_risk_pipeline --metrics-output reports/metrics.json

test:
	poetry run pytest

lint:
	poetry run ruff check src tests
	poetry run mypy src

format:
	poetry run ruff format src tests scripts
	poetry run ruff check --fix src tests scripts

clean:
	rm -rf data/bronze data/silver data/gold data/streaming/checkpoints models reports/metrics.json
