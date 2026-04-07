# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-04-06

### Added

**Collectors**
- `BaseCollector`: Abstract base class for pipeline run collectors with `collect(since)` interface.
- `GenericCollector`: Load pipeline runs from a `list[dict]`, `.json` array, or `.jsonl` file with Pydantic validation.
- `DbtCollector`: Parse dbt `run_results.json` files via configurable glob pattern. Maps model results to `PipelineRun` including duration from timing blocks and row counts from `adapter_response`.
- `AirflowCollector`: Query Airflow metadata DB (SQLite or PostgreSQL) via SQLAlchemy for `dag_run` + `task_instance` records. Conditional import guard for the `apache-airflow` dependency.

**Features**
- `FeatureRegistry`: Centrally tracks 11 features with metadata (name, formula, dtype, requires_history). Exposes `as_dataframe()` for inspection.
- `FeatureExtractor`: Extracts all 11 features from `PipelineRun` objects. Supports `fit()` / `transform()` / `fit_transform()` interface. Graceful degradation when historical data is insufficient.
- 11 canonical features: `duration_seconds`, `duration_z`, `rows_processed_log1p`, `row_count_delta_pct`, `null_rate_max`, `null_rate_delta`, `hour_of_day`, `day_of_week`, `is_weekend`, `status_is_success`, `rejection_rate`.

**Detectors**
- `AnomalyScore`: Dataclass capturing `run_id`, `pipeline_name`, `anomaly_score` [0,1], `is_anomaly`, `contributing_features`, `detector_name`, `timestamp`.
- `BaseDetector`: Abstract base class with `fit()`, `score()`, `batch_score()` interface and configurable `threshold`.
- `ZScoreDetector`: Rolling z-score detector with per-feature z-score computation and top-3 contributing feature selection.
- `IsolationForestDetector`: sklearn IsolationForest with 90/10 train/held-out split, permutation importance-based feature attribution, and min-max score normalisation.
- `EnsembleDetector`: Weighted average of any list of `BaseDetector` instances with automatic weight normalisation and deduplicated contributing features.

**Training**
- `Trainer`: Orchestrates data collection, optional pipeline filtering, and detector fitting. Returns a `FitResult` with training statistics.
- `ModelStore`: Filesystem-backed model store using `joblib.dump/load`. Saves JSON metadata sidecars alongside `.joblib` files. Supports `save()`, `load()`, `load_latest()`, `list_models()`.

**Scoring**
- `Scorer`: Scores runs, persists results to SQLite via SQLAlchemy, and routes alerts. Provides `score()`, `score_batch()`, `get_history()`.
- `AlertRouter`: Glob-pattern-based alert routing to Slack or log-only channels with configurable deduplication window.

**Integrations**
- `SlackIntegration`: Sends rich Slack Block Kit messages with score bar, contributing features, historical context, and anomaly badge.
- `AnomalyDetectionSensor`: Airflow `BaseSensorOperator` subclass (with stub fallback) that pokes for anomalies and raises `AirflowException` when detected.

**CLI** (`pad`)
- `pad collect`: Collect pipeline runs from dbt, Airflow, or generic sources.
- `pad train`: Train a detector on collected runs and save to model store.
- `pad score-batch`: Score a batch of runs from a JSONL file.
- `pad score`: Score a single run by ID.
- `pad explain`: Print a rich explanation panel for a scored run.
- `pad models list`: List all saved models in a model store directory.

**Tests**
- Unit tests: `FeatureExtractor` (10 tests), `IsolationForestDetector` (5 tests), `ZScoreDetector` (5 tests), `EnsembleDetector` (6 tests), `Scorer` (6 tests).
- Integration tests: `DbtCollector` (5 tests), full end-to-end pipeline (4 tests).

**Documentation**
- `docs/quickstart.md`: Five-command guide from installation to first detection.
- `docs/feature_reference.md`: Full feature table with formulas, dtypes, and descriptions.
- `docs/detector_comparison.md`: Pros/cons/when-to-use for each detector.
- `notebooks/exploration.ipynb`: Interactive exploration notebook with matplotlib timeline plot.

[0.1.0]: https://github.com/your-org/pipeline-anomaly-detector/releases/tag/v0.1.0
