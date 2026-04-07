# Feature Reference

The pipeline anomaly detector computes 11 features for each `PipelineRun`.
Features are organised into two groups: those that require historical run data
(windowed features) and those that can be computed from a single run alone.

---

## Feature Table

| Name | Formula | dtype | Requires History | Description |
|---|---|---|---|---|
| `duration_seconds` | `run.duration_seconds` | `float64` | No | Raw wall-clock duration of the run in seconds. |
| `duration_z` | `(duration - mean_30) / std_30` | `float64` | Yes | Z-score of duration vs the rolling 30-run per-pipeline window. High absolute values indicate unusually slow or fast runs. |
| `rows_processed_log1p` | `log1p(run.rows_processed)` | `float64` | No | Natural log-plus-one transform of row count. Reduces right skew and makes outlier row counts more detectable. |
| `row_count_delta_pct` | `(rows - prev_rows) / prev_rows * 100` | `float64` | Yes | Percentage change in row count versus the immediately preceding run of the same pipeline. |
| `null_rate_max` | `max(run.null_rate.values())` | `float64` | No | Maximum null rate across all tracked columns. Zero when `null_rate` is empty. |
| `null_rate_delta` | `null_rate_max - prev_null_rate_max` | `float64` | Yes | Change in `null_rate_max` versus the previous run. Positive values indicate a data quality degradation. |
| `hour_of_day` | `run.start_time.hour` | `int8` | No | UTC hour (0–23) when the run started. Useful for detecting runs at unexpected times. |
| `day_of_week` | `run.start_time.weekday()` | `int8` | No | Day of week (0 = Monday, 6 = Sunday). Captures weekly seasonality patterns. |
| `is_weekend` | `1 if weekday >= 5 else 0` | `int8` | No | Binary flag: 1 for Saturday or Sunday, 0 otherwise. |
| `status_is_success` | `1 if run.status == 'success' else 0` | `int8` | No | Binary flag indicating a successful run. Failed or skipped runs score 0. |
| `rejection_rate` | `rows_rejected / rows_processed` | `float64` | No | Fraction of rows rejected. Returns 0.0 when `rows_processed == 0` to avoid division by zero. |

---

## Graceful Degradation for History-Dependent Features

Features marked **Requires History = Yes** fall back gracefully when
insufficient historical data is available:

- **`duration_z`**: Returns `0.0` when fewer than 2 historical runs exist for
  the pipeline (standard deviation is undefined).
- **`row_count_delta_pct`**: Returns `0.0` when no prior run exists.
- **`null_rate_delta`**: Returns `0.0` when no prior run exists.

The rolling window defaults to **30 runs**. When fewer than 30 runs are
available, all available runs are used — no error is raised.

---

## Feature Extractor API

```python
from pipeline_anomaly_detector.features import FeatureExtractor

extractor = FeatureExtractor(window=30)
extractor.fit(historical_runs)          # store history for windowed features
df = extractor.transform(new_runs)      # returns pd.DataFrame, index=run_id

# Or in one step:
df = extractor.fit_transform(all_runs)
```

## Feature Registry

```python
from pipeline_anomaly_detector.features import FEATURE_REGISTRY

# Print all features as a DataFrame
print(FEATURE_REGISTRY.as_dataframe())

# Access a specific feature spec
spec = FEATURE_REGISTRY.get("duration_z")
print(spec.requires_history)  # True
```
