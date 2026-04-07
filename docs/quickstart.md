# Quickstart

Get from zero to your first detected anomaly in five commands.

## Prerequisites

- Python 3.11+
- A dbt project with `run_results.json` **or** a JSONL file of pipeline runs

## Installation

```bash
pip install pipeline-anomaly-detector
```

For development (tests, notebook):

```bash
pip install "pipeline-anomaly-detector[dev]"
```

---

## Step 1 — Collect pipeline runs

**From a dbt project:**

```bash
pad collect \
  --source dbt \
  --dbt-dir "./target/run_results.json" \
  --since 2024-01-01 \
  --output runs.jsonl
```

**From a generic JSONL file:**

```bash
pad collect \
  --source generic \
  --input my_pipeline_runs.json \
  --output runs.jsonl
```

**From Airflow:**

```bash
pad collect \
  --source airflow \
  --airflow-db "sqlite:///~/airflow/airflow.db" \
  --since 2024-01-01 \
  --output runs.jsonl
```

---

## Step 2 — Train an anomaly detector

```bash
pad train \
  --input runs.jsonl \
  --detector ensemble \
  --output ./models
```

This trains an `EnsembleDetector` (ZScore + IsolationForest) on your collected
runs and saves the model to `./models/`.

Available detector types:

| Value | Description |
|---|---|
| `ensemble` | Weighted combination of ZScore + IsolationForest (recommended) |
| `zscore` | Fast, interpretable z-score baseline |
| `isolation_forest` | sklearn IsolationForest, handles non-linear anomalies |

---

## Step 3 — Score a batch of new runs

```bash
pad score-batch \
  --input new_runs.jsonl \
  --model ./models/global_ensemble_20240115T120000Z.joblib \
  --db scores.db
```

The results are printed to the terminal and persisted to `scores.db`.

---

## Step 4 — Explain an anomaly

```bash
pad explain \
  --run-id anomaly_duration_000 \
  --model ./models/global_ensemble_20240115T120000Z.joblib \
  --db scores.db
```

This prints a Rich panel with the anomaly score bar, contributing features,
and is_anomaly status.

---

## Step 5 — List saved models

```bash
pad models list --store-dir ./models
```

---

## What's next?

- See [Feature Reference](feature_reference.md) for the full feature list.
- See [Detector Comparison](detector_comparison.md) to pick the right detector.
- Configure Slack alerts by setting `SLACK_WEBHOOK_URL` in your environment.
