"""Feature registry metadata for generated feature tables."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeatureDefinition:
    """Metadata for one generated feature."""

    name: str
    group: str
    dtype: str
    description: str
    source_columns: list[str]
    leakage_risk: str
    default_value: str | int | float | None
    owner: str


def _feature_definition(
    name: str,
    group: str,
    dtype: str,
    description: str,
    source_columns: list[str],
    leakage_risk: str = "low",
    default_value: str | int | float | None = 0,
    owner: str = "risk-ml",
) -> FeatureDefinition:
    """Create a compact feature definition entry."""
    return FeatureDefinition(
        name=name,
        group=group,
        dtype=dtype,
        description=description,
        source_columns=source_columns,
        leakage_risk=leakage_risk,
        default_value=default_value,
        owner=owner,
    )


def get_feature_registry() -> list[FeatureDefinition]:
    """Return the feature registry for the main feature pipeline."""
    registry = [
        _feature_definition("amount_log1p", "transaction", "double", "Natural log of amount plus one.", ["amount"]),
        _feature_definition(
            "origin_balance_delta",
            "transaction",
            "double",
            "Origin account balance delta across the transaction.",
            ["oldbalanceOrg", "newbalanceOrig"],
        ),
        _feature_definition(
            "destination_balance_delta",
            "transaction",
            "double",
            "Destination account balance delta across the transaction.",
            ["oldbalanceDest", "newbalanceDest"],
        ),
        _feature_definition(
            "origin_delta_minus_amount",
            "transaction",
            "double",
            "Origin balance delta relative to stated amount.",
            ["oldbalanceOrg", "newbalanceOrig", "amount"],
        ),
        _feature_definition(
            "destination_delta_minus_amount",
            "transaction",
            "double",
            "Destination balance delta relative to stated amount.",
            ["oldbalanceDest", "newbalanceDest", "amount"],
        ),
        _feature_definition("origin_balance_was_zero", "transaction", "int", "Origin balance was zero before the transaction.", ["oldbalanceOrg"]),
        _feature_definition("origin_balance_is_zero", "transaction", "int", "Origin balance is zero after the transaction.", ["newbalanceOrig"]),
        _feature_definition("destination_balance_was_zero", "transaction", "int", "Destination balance was zero before the transaction.", ["oldbalanceDest"]),
        _feature_definition("destination_balance_is_zero", "transaction", "int", "Destination balance is zero after the transaction.", ["newbalanceDest"]),
        _feature_definition("is_cash_out", "transaction", "int", "Transaction type is CASH_OUT.", ["type"]),
        _feature_definition("is_transfer", "transaction", "int", "Transaction type is TRANSFER.", ["type"]),
        _feature_definition("is_payment", "transaction", "int", "Transaction type is PAYMENT.", ["type"]),
        _feature_definition("is_debit", "transaction", "int", "Transaction type is DEBIT.", ["type"]),
        _feature_definition("is_cash_in", "transaction", "int", "Transaction type is CASH_IN.", ["type"]),
        _feature_definition("is_merchant_destination", "transaction", "int", "Destination account name appears to be a merchant.", ["nameDest"]),
        _feature_definition("large_amount_flag", "transaction", "int", "Amount exceeds the deterministic large-amount threshold.", ["amount"]),
        _feature_definition("previous_step_by_origin", "temporal", "bigint", "Previous observed transaction step for the origin entity.", ["nameOrig", "step"], default_value=None),
        _feature_definition("steps_since_previous_origin_tx", "temporal", "bigint", "Elapsed steps since the previous origin transaction.", ["nameOrig", "step"], default_value=-1),
        _feature_definition("origin_tx_count_before", "temporal", "bigint", "Count of earlier transactions by the origin entity.", ["nameOrig", "step"], default_value=0),
        _feature_definition("origin_amount_mean_before", "temporal", "double", "Mean prior transaction amount for the origin entity.", ["nameOrig", "step", "amount"], default_value=0.0),
        _feature_definition("origin_amount_std_before", "temporal", "double", "Population standard deviation of prior origin amounts.", ["nameOrig", "step", "amount"], default_value=0.0),
        _feature_definition("origin_amount_to_mean_ratio", "temporal", "double", "Current amount divided by the prior origin mean amount.", ["nameOrig", "step", "amount"], default_value=0.0),
        _feature_definition("origin_amount_zscore_before", "temporal", "double", "Current amount z-score against prior origin history.", ["nameOrig", "step", "amount"], default_value=0.0),
        _feature_definition("origin_total_tx_count", "entity", "bigint", "Total transactions observed for the origin entity.", ["nameOrig"], default_value=0),
        _feature_definition("origin_avg_amount", "entity", "double", "Average amount sent by the origin entity.", ["nameOrig", "amount"], default_value=0.0),
        _feature_definition("origin_max_amount", "entity", "double", "Maximum amount sent by the origin entity.", ["nameOrig", "amount"], default_value=0.0),
        _feature_definition("origin_unique_destinations", "entity", "bigint", "Distinct destinations used by the origin entity.", ["nameOrig", "nameDest"], default_value=0),
        _feature_definition("destination_total_tx_count", "entity", "bigint", "Total transactions received by the destination entity.", ["nameDest"], default_value=0),
        _feature_definition("destination_avg_amount", "entity", "double", "Average amount received by the destination entity.", ["nameDest", "amount"], default_value=0.0),
        _feature_definition("destination_max_amount", "entity", "double", "Maximum amount received by the destination entity.", ["nameDest", "amount"], default_value=0.0),
        _feature_definition("destination_unique_origins", "entity", "bigint", "Distinct originators that have sent to the destination entity.", ["nameOrig", "nameDest"], default_value=0),
        _feature_definition("origin_destination_pair_count", "entity", "bigint", "Observed transaction count for the origin-destination pair.", ["nameOrig", "nameDest"], default_value=0),
        _feature_definition("origin_destination_avg_amount", "entity", "double", "Average amount for the origin-destination pair.", ["nameOrig", "nameDest", "amount"], default_value=0.0),
        _feature_definition("origin_out_degree", "graph", "bigint", "Count of distinct destination nodes connected to the origin node.", ["nameOrig", "nameDest"], default_value=0),
        _feature_definition("destination_in_degree", "graph", "bigint", "Count of distinct origin nodes connected to the destination node.", ["nameOrig", "nameDest"], default_value=0),
        _feature_definition("edge_frequency", "graph", "bigint", "Observed frequency of the origin-destination edge.", ["nameOrig", "nameDest"], default_value=0),
        _feature_definition(
            "destination_historical_fraud_rate",
            "graph",
            "double",
            "Historical label rate for the destination node.",
            ["nameDest", "isFraud"],
            leakage_risk="medium",
            default_value=0.0,
        ),
        _feature_definition(
            "destination_historical_fraud_count",
            "graph",
            "bigint",
            "Historical fraud count for the destination node.",
            ["nameDest", "isFraud"],
            leakage_risk="medium",
            default_value=0,
        ),
        _feature_definition("has_card1", "identity", "int", "card1 field is present.", ["card1"]),
        _feature_definition("has_card2", "identity", "int", "card2 field is present.", ["card2"]),
        _feature_definition("has_card3", "identity", "int", "card3 field is present.", ["card3"]),
        _feature_definition("has_card4", "identity", "int", "card4 field is present.", ["card4"]),
        _feature_definition("has_card5", "identity", "int", "card5 field is present.", ["card5"]),
        _feature_definition("has_card6", "identity", "int", "card6 field is present.", ["card6"]),
        _feature_definition("p_emaildomain_normalized", "identity", "string", "Normalized purchaser email domain.", ["P_emaildomain"], default_value="unknown"),
        _feature_definition("r_emaildomain_normalized", "identity", "string", "Normalized recipient email domain.", ["R_emaildomain"], default_value="unknown"),
        _feature_definition("has_p_emaildomain", "identity", "int", "Purchaser email domain is present.", ["P_emaildomain"]),
        _feature_definition("has_r_emaildomain", "identity", "int", "Recipient email domain is present.", ["R_emaildomain"]),
        _feature_definition("email_domains_match", "identity", "int", "Purchaser and recipient email domains match after normalization.", ["P_emaildomain", "R_emaildomain"]),
        _feature_definition("device_type_normalized", "identity", "string", "Normalized IEEE-CIS device type.", ["DeviceType"], default_value="unknown"),
        _feature_definition("device_info_normalized", "identity", "string", "Normalized IEEE-CIS device info string.", ["DeviceInfo"], default_value="unknown"),
        _feature_definition("identity_missing_value_count", "identity", "int", "Count of missing values across present identity-related columns.", ["id_*", "card*", "addr*", "dist*", "M*", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]),
        _feature_definition("product_cd_avg_transaction_amount", "identity", "double", "Average transaction amount for the product code.", ["ProductCD", "TransactionAmt"], default_value=0.0),
        _feature_definition("product_cd_max_transaction_amount", "identity", "double", "Maximum transaction amount for the product code.", ["ProductCD", "TransactionAmt"], default_value=0.0),
        _feature_definition("product_cd_transaction_count", "identity", "bigint", "Transaction count observed for the product code.", ["ProductCD"], default_value=0),
    ]
    return registry


def feature_registry_to_markdown(
    feature_registry: list[FeatureDefinition] | None = None,
) -> str:
    """Render the feature registry as GitHub-friendly Markdown."""
    registry = feature_registry or get_feature_registry()
    lines = [
        "# Feature Registry",
        "",
        "| Name | Group | Type | Source Columns | Leakage Risk | Default | Owner | Description |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in registry:
        source_columns = ", ".join(entry.source_columns)
        default_value = "" if entry.default_value is None else str(entry.default_value)
        lines.append(
            f"| {entry.name} | {entry.group} | {entry.dtype} | {source_columns} | "
            f"{entry.leakage_risk} | {default_value} | {entry.owner} | {entry.description} |"
        )
    return "\n".join(lines) + "\n"


def feature_registry_to_json(
    feature_registry: list[FeatureDefinition] | None = None,
) -> str:
    """Render the feature registry as formatted JSON."""
    registry = feature_registry or get_feature_registry()
    return json.dumps([asdict(entry) for entry in registry], indent=2)
