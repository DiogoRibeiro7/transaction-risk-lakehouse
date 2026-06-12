# Limitations

This project is a portfolio demonstration of production-style fraud detection engineering. These are the honest boundaries of what it shows.

## Synthetic data

The default dataset is PaySim (or a locally generated PaySim-like sample). Synthetic simulations do not reproduce real fraud behaviour: there is no adversary adapting to the model, fraud typologies are simplistic, and feature distributions are cleaner than production payment data. Metrics computed on it demonstrate that the pipeline works, not that the model would perform at this level in production. The IEEE-CIS path uses real (anonymized) data but is a one-off competition snapshot, not a live feed.

## Label leakage risk

- `destination_historical_fraud_rate` aggregates the fraud label over destination accounts. The feature pipeline computes it over the full input table; in a real system it must be derived from confirmed-fraud history available *before* each transaction, with realistic confirmation delay.
- Any feature built from the silver table after labels are attached carries similar risk. The feature registry flags leakage risk per feature; review those notes before trusting an offline metric.

## Temporal validation assumptions

The temporal split assumes `step` (PaySim) or the `TransactionDT`-derived event time (IEEE-CIS) is a faithful, complete ordering of events. Real pipelines deal with late-arriving events, clock skew across systems, and backfills, none of which are modelled here. The split boundaries are also fixed fractions rather than business-calendar windows.

## Operational threshold assumptions

- Alert-rate thresholding assumes review capacity is constant over time.
- Expected-value thresholding treats fraud loss rate, review cost, and recovery rate as known constants supplied by the operator. In practice these vary by segment, evolve over time, and are themselves uncertain estimates.
- Threshold selection happens on a single validation window; production systems should re-estimate thresholds on a schedule.

## Model monitoring limitations

- PSI with quantile bins is a coarse drift signal; it misses correlated drift across features and can be noisy on small windows.
- Labeled monitoring (precision/recall by bucket, label-rate drift) assumes labels arrive in time to join against the monitoring window. Real fraud labels lag days to months; the reports distinguish labeled from unlabeled mode but do not model label delay itself.
- There is no alerting/paging integration; reports are static JSON/Markdown artifacts.

## Engineering scope

- All demos run on a single machine with `local[*]` Spark. Cluster deployment, autoscaling, secrets management, and access control are out of scope.
- The model registry is a local JSON Lines file, not a concurrent multi-user service.
- The streaming demo uses the file source and `foreachBatch`; production streaming needs a durable broker (Kafka), exactly-once sinks, and schema evolution handling.
