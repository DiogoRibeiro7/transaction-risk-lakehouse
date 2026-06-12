"""Generate small demo figures for the README from the sample pipeline outputs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_FIGURES_DIR = Path("reports/figures")


def _matplotlib():
    """Import matplotlib lazily with a non-interactive backend."""
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for demo artifacts. Install dev dependencies with `poetry install`."
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def build_class_imbalance_figure(label_counts: dict[int, int], output_path: Path) -> Path:
    """Plot fraud vs non-fraud transaction counts on a log scale."""
    plt = _matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(label) for label in sorted(label_counts)]
    counts = [label_counts[label] for label in sorted(label_counts)]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, counts, color=["tab:blue", "tab:red"][: len(labels)])
    ax.set_yscale("log")
    ax.set_xlabel("isFraud")
    ax.set_ylabel("transactions (log scale)")
    ax.set_title("Class imbalance")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def build_probability_distribution_figure(
    bin_edges: list[float],
    bin_counts: list[int],
    output_path: Path,
) -> Path:
    """Plot the fraud probability histogram of scored transactions."""
    if len(bin_edges) != len(bin_counts) + 1:
        raise ValueError("bin_edges must have exactly one more element than bin_counts.")
    plt = _matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    widths = [bin_edges[i + 1] - bin_edges[i] for i in range(len(bin_counts))]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(bin_edges[:-1], bin_counts, width=widths, align="edge", color="tab:purple")
    ax.set_yscale("log")
    ax.set_xlabel("fraud probability")
    ax.set_ylabel("transactions (log scale)")
    ax.set_title("Fraud probability distribution")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def build_alert_threshold_figure(
    thresholds: list[float],
    alert_counts: list[int],
    output_path: Path,
) -> Path:
    """Plot alert volume against candidate thresholds."""
    if len(thresholds) != len(alert_counts):
        raise ValueError("thresholds and alert_counts must have the same length.")
    plt = _matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(thresholds, alert_counts, marker="o", color="tab:orange")
    ax.set_xlabel("probability threshold")
    ax.set_ylabel("alerts")
    ax.set_title("Alert volume by threshold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def build_metric_comparison_figure(results: list[dict], output_path: Path) -> Path:
    """Plot benchmark metrics per model type as grouped bars."""
    if not results:
        raise ValueError("results must not be empty.")
    plt = _matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_names = ["roc_auc", "pr_auc", "precision_at_k", "recall_at_k"]
    model_types = [result["model_type"] for result in results]
    bar_width = 0.8 / len(metric_names)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for index, metric in enumerate(metric_names):
        positions = [base + index * bar_width for base in range(len(model_types))]
        values = [float(result.get(metric, 0.0)) for result in results]
        ax.bar(positions, values, width=bar_width, label=metric)
    ax.set_xticks([base + 0.4 - bar_width / 2 for base in range(len(model_types))])
    ax.set_xticklabels(model_types, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title("Model benchmark comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def maybe_create_gif(figure_paths: list[Path], output_path: Path) -> bool:
    """Assemble figures into a GIF when imageio is available."""
    try:
        import imageio.v3 as imageio
    except ImportError:
        logger.warning("imageio is not installed; skipping GIF generation.")
        return False

    frames = [imageio.imread(path) for path in figure_paths if path.exists()]
    if not frames:
        logger.warning("No figures available for GIF generation.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, frames, duration=1500, loop=0)
    logger.info("Wrote demo GIF to %s", output_path)
    return True


def generate_artifacts(
    silver_path: str,
    scored_path: str,
    benchmark_metrics_path: str,
    figures_dir: Path,
    spark_config: str,
    with_gif: bool = False,
) -> list[Path]:
    """Aggregate pipeline outputs with Spark and render all available figures."""
    import json

    from pyspark.sql import functions as F

    from transaction_risk.spark.io import read_table
    from transaction_risk.spark.session import create_spark_session_from_yaml

    spark = create_spark_session_from_yaml(spark_config)
    figures: list[Path] = []

    transactions = read_table(spark, silver_path)
    label_counts = {
        int(row["isFraud"]): int(row["count"])
        for row in transactions.groupBy("isFraud").count().collect()
    }
    figures.append(
        build_class_imbalance_figure(label_counts, figures_dir / "class_imbalance.png")
    )

    scored_dir = Path(scored_path)
    if scored_dir.exists():
        scored = read_table(spark, scored_path)
        bins = 20
        histogram = (
            scored.withColumn(
                "bucket",
                F.least(F.floor(F.col("fraud_probability") * bins).cast("int"), F.lit(bins - 1)),
            )
            .groupBy("bucket")
            .count()
            .collect()
        )
        bucket_counts = {int(row["bucket"]): int(row["count"]) for row in histogram}
        bin_counts = [bucket_counts.get(bucket, 0) for bucket in range(bins)]
        bin_edges = [bucket / bins for bucket in range(bins + 1)]
        figures.append(
            build_probability_distribution_figure(
                bin_edges, bin_counts, figures_dir / "fraud_probability_distribution.png"
            )
        )

        thresholds = [round(0.05 * step, 2) for step in range(1, 20)]
        alert_counts = [
            int(scored.filter(F.col("fraud_probability") >= threshold).count())
            for threshold in thresholds
        ]
        figures.append(
            build_alert_threshold_figure(
                thresholds, alert_counts, figures_dir / "alert_threshold.png"
            )
        )
    else:
        logger.warning("Scored output %s not found; skipping score figures.", scored_path)

    benchmark_path = Path(benchmark_metrics_path)
    if benchmark_path.exists():
        results = json.loads(benchmark_path.read_text(encoding="utf-8"))
        figures.append(
            build_metric_comparison_figure(results, figures_dir / "model_metric_comparison.png")
        )
    else:
        logger.warning(
            "Benchmark metrics %s not found; skipping comparison figure. Run `make benchmark` first.",
            benchmark_path,
        )

    spark.stop()

    if with_gif:
        maybe_create_gif(figures, figures_dir / "demo.gif")
    return figures


def main() -> None:
    """Run the demo artifact generator."""
    parser = argparse.ArgumentParser(description="Generate demo figures from sample pipeline outputs")
    parser.add_argument("--silver", default="data/silver/transactions")
    parser.add_argument("--scored", default="data/scored/batch")
    parser.add_argument("--benchmark-metrics", default="reports/benchmark/metrics.json")
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--spark-config", default="conf/spark.local.yaml")
    parser.add_argument("--gif", action="store_true", help="Also assemble a GIF (requires imageio)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    figures = generate_artifacts(
        silver_path=args.silver,
        scored_path=args.scored,
        benchmark_metrics_path=args.benchmark_metrics,
        figures_dir=Path(args.figures_dir),
        spark_config=args.spark_config,
        with_gif=args.gif,
    )
    for figure in figures:
        logger.info("Ready: %s", figure)


if __name__ == "__main__":
    main()
