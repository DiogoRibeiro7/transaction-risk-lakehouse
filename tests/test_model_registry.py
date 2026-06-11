from __future__ import annotations

from pathlib import Path

from transaction_risk.models.registry import (
    ModelMetric,
    get_latest_model,
    load_registry,
    register_model,
)


def _register(registry_path: Path, model_type: str = "logistic_regression", notes: str | None = None):
    return register_model(
        model_path="models/fraud_risk_pipeline",
        registry_path=registry_path,
        metrics={"roc_auc": 0.91, "pr_auc": 0.42},
        feature_table_path="data/gold/features",
        model_type=model_type,
        threshold=0.35,
        notes=notes,
    )


def test_register_model_appends_versions(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"

    first = _register(registry_path)
    second = _register(registry_path, model_type="gbt", notes="candidate")

    assert first.version == 1
    assert second.version == 2
    assert registry_path.read_text(encoding="utf-8").count("\n") == 2


def test_load_registry_round_trips_entries(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"
    _register(registry_path)
    _register(registry_path, model_type="gbt")

    entries = load_registry(registry_path)

    assert len(entries) == 2
    assert entries[0].version == 1
    assert entries[0].model_type == "logistic_regression"
    assert entries[0].threshold == 0.35
    assert ModelMetric(name="roc_auc", value=0.91) in entries[0].metrics
    assert entries[1].model_type == "gbt"


def test_load_registry_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert load_registry(tmp_path / "missing.jsonl") == []


def test_get_latest_model_selects_highest_version(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.jsonl"

    assert get_latest_model(registry_path) is None

    _register(registry_path)
    latest = _register(registry_path, model_type="random_forest")

    selected = get_latest_model(registry_path)
    assert selected is not None
    assert selected.version == latest.version
    assert selected.model_type == "random_forest"
