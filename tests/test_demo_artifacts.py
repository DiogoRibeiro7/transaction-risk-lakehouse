from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path("scripts") / "generate_demo_artifacts.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("generate_demo_artifacts", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_class_imbalance_figure_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    module = _load_script_module()

    output = module.build_class_imbalance_figure({0: 990, 1: 10}, tmp_path / "imbalance.png")

    assert output.exists()
    assert output.stat().st_size > 0


def test_probability_distribution_figure_validates_bins(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    module = _load_script_module()

    with pytest.raises(ValueError, match="one more element"):
        module.build_probability_distribution_figure([0.0, 0.5], [1, 2], tmp_path / "dist.png")

    output = module.build_probability_distribution_figure(
        [0.0, 0.5, 1.0], [90, 10], tmp_path / "dist.png"
    )
    assert output.exists()


def test_alert_threshold_figure_validates_lengths(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    module = _load_script_module()

    with pytest.raises(ValueError, match="same length"):
        module.build_alert_threshold_figure([0.1, 0.2], [5], tmp_path / "alerts.png")

    output = module.build_alert_threshold_figure([0.1, 0.5, 0.9], [80, 20, 3], tmp_path / "alerts.png")
    assert output.exists()


def test_metric_comparison_figure_rejects_empty_results(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    module = _load_script_module()

    with pytest.raises(ValueError, match="must not be empty"):
        module.build_metric_comparison_figure([], tmp_path / "comparison.png")

    output = module.build_metric_comparison_figure(
        [
            {"model_type": "logistic_regression", "roc_auc": 0.9, "pr_auc": 0.4, "precision_at_k": 0.3, "recall_at_k": 0.5},
            {"model_type": "gbt", "roc_auc": 0.95, "pr_auc": 0.5, "precision_at_k": 0.4, "recall_at_k": 0.6},
        ],
        tmp_path / "comparison.png",
    )
    assert output.exists()


def test_maybe_create_gif_handles_missing_imageio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("imageio"):
            raise ImportError("imageio missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    created = module.maybe_create_gif([tmp_path / "missing.png"], tmp_path / "demo.gif")

    assert created is False
    assert not (tmp_path / "demo.gif").exists()
