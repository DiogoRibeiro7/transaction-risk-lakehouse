"""Generate a small PaySim-like dataset for local demos and tests."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]


def generate_row(index: int, fraud_rate: float) -> dict[str, str | int | float]:
    """Generate a synthetic PaySim-like transaction row."""
    is_fraud = random.random() < fraud_rate
    tx_type = random.choices(
        TRANSACTION_TYPES,
        weights=[0.42, 0.16 if not is_fraud else 0.45, 0.22 if not is_fraud else 0.45, 0.08, 0.12],
        k=1,
    )[0]

    origin = f"C{random.randint(1, 1200):07d}"
    destination_prefix = "M" if tx_type == "PAYMENT" else "C"
    destination = f"{destination_prefix}{random.randint(1, 1800):07d}"

    old_origin = round(random.uniform(50.0, 250_000.0), 2)
    if is_fraud:
        amount = round(random.uniform(max(10.0, old_origin * 0.65), max(20.0, old_origin * 1.05)), 2)
    else:
        amount = round(random.expovariate(1 / 5000.0) + random.uniform(0, 500), 2)

    new_origin = max(old_origin - amount, 0.0)
    old_dest = round(random.uniform(0.0, 350_000.0), 2)
    new_dest = old_dest + amount if tx_type in {"TRANSFER", "CASH_OUT"} else old_dest

    return {
        "step": index // 20,
        "type": tx_type,
        "amount": round(amount, 2),
        "nameOrig": origin,
        "oldbalanceOrg": round(old_origin, 2),
        "newbalanceOrig": round(new_origin, 2),
        "nameDest": destination,
        "oldbalanceDest": round(old_dest, 2),
        "newbalanceDest": round(new_dest, 2),
        "isFraud": int(is_fraud),
        "isFlaggedFraud": int(is_fraud and amount > 200_000),
    }


def generate_sample(output: Path, rows: int, fraud_rate: float, seed: int = 42) -> None:
    """Generate and write synthetic PaySim-like data."""
    if rows <= 0:
        raise ValueError("rows must be positive.")
    if not 0 <= fraud_rate <= 1:
        raise ValueError("fraud_rate must be between 0 and 1.")

    random.seed(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    ]

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(rows):
            writer.writerow(generate_row(index=index, fraud_rate=fraud_rate))


def main() -> None:
    """Run the sample data generator."""
    parser = argparse.ArgumentParser(description="Generate a PaySim-like sample dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--fraud-rate", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_sample(Path(args.output), rows=args.rows, fraud_rate=args.fraud_rate, seed=args.seed)


if __name__ == "__main__":
    main()
