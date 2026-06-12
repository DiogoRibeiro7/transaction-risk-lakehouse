# Figures

Figures in this folder are generated locally and are not committed (the folder is git-ignored except for this README).

To regenerate them, run the sample pipeline and the demo artifact script:

```bash
make sample-data
make ingest
make features
make train
make demo-artifacts
```

`make demo-artifacts` runs `scripts/generate_demo_artifacts.py`, which produces:

- `class_imbalance.png` — fraud vs non-fraud transaction counts
- `fraud_probability_distribution.png` — score distribution of the trained model
- `alert_threshold.png` — alert volume against candidate thresholds
- `model_metric_comparison.png` — benchmark metric comparison (only when `reports/benchmark/metrics.json` exists; run `make benchmark` first)

The script uses matplotlib only. GIF assembly is optional and requires `imageio`; when it is not installed the script logs a warning and still writes the static PNGs.
