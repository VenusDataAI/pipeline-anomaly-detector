# Detector Comparison

The pipeline anomaly detector ships three detection algorithms and an ensemble
wrapper. This page compares them to help you choose the right one for your use
case.

---

## Overview Table

| Property | ZScoreDetector | IsolationForestDetector | EnsembleDetector |
|---|---|---|---|
| **Detector name** | `zscore` | `isolation_forest` | `ensemble` |
| **Algorithm** | Rolling z-score per feature | sklearn IsolationForest | Weighted average of sub-detectors |
| **Default threshold** | 0.5 | 0.5 | 0.6 |
| **Requires training data** | Yes | Yes | Yes (delegates to sub-detectors) |
| **Handles non-linear anomalies** | No | Yes | Yes (via IsolationForest) |
| **Interpretability** | High — z-score per feature | Medium — permutation importance | Medium — union of sub-features |
| **Training speed** | Fast (O(n)) | Moderate (O(n log n)) | Depends on sub-detectors |
| **Minimum recommended runs** | 5 (degrades gracefully) | 20 | 20 |
| **Contamination parameter** | — | Yes (`contamination=0.05`) | — |
| **CLI flag** | `zscore` | `isolation_forest` | `ensemble` |

---

## ZScoreDetector

### How it works

For each scored run the detector:

1. Extracts all 11 features.
2. Computes per-feature z-scores against the rolling window of the last 30
   runs for the same pipeline.
3. Maps `max(|z|) / 10.0` to `anomaly_score` (capped at 1.0).
4. Flags features with `|z| > 2.0` as `contributing_features`.

### Pros

- Extremely fast and deterministic.
- Highly interpretable: you can see exactly which feature caused the anomaly
  and by how many standard deviations.
- Works well with limited data (degrades to `score=0` rather than erroring).
- No hyperparameter tuning required beyond `window`.

### Cons

- Assumes normality within each feature distribution.
- Cannot detect anomalies that emerge from *combinations* of features (each
  feature is evaluated independently).
- Sensitive to outliers in the training window (one bad run distorts the mean
  and std).

### When to use

- You want fast, explainable detections with clear thresholds.
- Your anomalies manifest as single-feature spikes (duration, row count, null
  rate).
- You are just getting started and want a simple baseline.

```python
from pipeline_anomaly_detector.models import ZScoreDetector

detector = ZScoreDetector(window=30, threshold=0.5)
detector.fit(historical_runs)
score = detector.score(new_run)
```

---

## IsolationForestDetector

### How it works

1. Extracts all 11 features from training runs.
2. Splits 90/10 into train and held-out sets.
3. Fits sklearn's `IsolationForest` on the training split.
4. Computes permutation importance on the held-out set to rank features.
5. At score time, maps `decision_function` output to [0, 1] via min-max
   normalisation (inverted so higher = more anomalous).

### Pros

- Detects multi-variate anomalies that no single feature would catch alone.
- Non-parametric — no assumption of normality.
- Scales well to large feature spaces and many pipelines.
- Feature importances provide post-hoc interpretability.

### Cons

- Scores are less interpretable than z-scores in absolute terms.
- Requires more training data for reliable results (~50+ runs recommended).
- The `contamination` hyperparameter must be set thoughtfully.
- Non-deterministic unless `random_state` is fixed.

### When to use

- You have complex pipelines with correlated features.
- You want to catch subtle, multi-dimensional anomalies.
- You have at least 50 clean training runs per pipeline.

```python
from pipeline_anomaly_detector.models import IsolationForestDetector

detector = IsolationForestDetector(contamination=0.05, threshold=0.5)
detector.fit(historical_runs)
score = detector.score(new_run)
```

---

## EnsembleDetector

### How it works

The ensemble combines the scores of any number of sub-detectors via a weighted
average:

```
anomaly_score = Σ(weight_i * sub_score_i)
```

Weights are normalised to sum to 1.0. Contributing features are the
deduplicated union of all sub-detector contributing features.

### Pros

- Reduces false positives from any single detector.
- Benefits from the complementary strengths of ZScore (interpretability) and
  IsolationForest (multi-variate detection).
- Fully configurable: add any `BaseDetector` subclass as a sub-detector.
- Higher default threshold (0.6) reduces alert fatigue.

### Cons

- Slightly slower than individual detectors (runs both).
- Slightly harder to attribute a single anomaly cause (two detectors may
  disagree).

### When to use

- **Production workloads** — the recommended default.
- When you want the best balance of precision and recall.
- When false positives are costly (raise the threshold to 0.7+).
- When you want to combine domain-specific custom detectors.

```python
from pipeline_anomaly_detector.models import (
    EnsembleDetector, ZScoreDetector, IsolationForestDetector
)

detector = EnsembleDetector(
    detectors=[ZScoreDetector(window=30), IsolationForestDetector(contamination=0.05)],
    weights=None,   # equal weights
    threshold=0.6,
)
detector.fit(historical_runs)
score = detector.score(new_run)
```

---

## Threshold Tuning

All detectors expose a `threshold` parameter (default varies). The anomaly
decision is:

```
is_anomaly = anomaly_score >= threshold
```

| Threshold | Effect |
|---|---|
| Low (0.3) | More sensitive — catches subtle anomalies, more false positives |
| Medium (0.5) | Balanced — good starting point |
| High (0.7) | Conservative — only flags severe anomalies |

Tune the threshold using a labelled validation set or business domain knowledge
about acceptable false-positive rates.
