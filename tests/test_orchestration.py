from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PIPELINE_PATH = Path("orchestration") / "dagster" / "transaction_risk_pipeline.py"


def _load_pipeline_module():
    spec = importlib.util.spec_from_file_location("transaction_risk_pipeline", PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pipeline_module_imports_without_dagster_requirement() -> None:
    module = _load_pipeline_module()

    assert hasattr(module, "PIPELINE_STEPS")
    assert hasattr(module, "build_step_command")
    assert hasattr(module, "run_step")


def test_pipeline_steps_cover_full_workflow() -> None:
    module = _load_pipeline_module()

    expected_steps = {
        "generate_sample_data",
        "ingest_raw_data",
        "build_features",
        "train_model",
        "score_batch",
        "monitoring_report",
    }
    assert set(module.PIPELINE_STEPS) == expected_steps


def test_pipeline_commands_call_project_cli() -> None:
    module = _load_pipeline_module()

    for step_name, command in module.PIPELINE_STEPS.items():
        assert command[:2] == ["poetry", "run"], step_name
        assert "transaction-risk" in command or "python" in command, step_name


def test_build_step_command_returns_copy() -> None:
    module = _load_pipeline_module()

    command = module.build_step_command("train_model")
    command.append("--mutated")

    assert "--mutated" not in module.PIPELINE_STEPS["train_model"]


def test_build_step_command_rejects_unknown_step() -> None:
    module = _load_pipeline_module()

    with pytest.raises(ValueError, match="Unknown pipeline step"):
        module.build_step_command("not_a_step")


def test_dagster_job_is_defined_when_dagster_available() -> None:
    pytest.importorskip("dagster")
    module = _load_pipeline_module()

    assert module.HAS_DAGSTER is True
    assert hasattr(module, "transaction_risk_job")
