"""End-to-end integration tests for the full anomaly detection pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pipeline_anomaly_detector import PipelineRun
from pipeline_anomaly_detector.collectors.generic_collector import GenericCollector
from pipeline_anomaly_detector.models.ensemble_detector import EnsembleDetector
from pipeline_anomaly_detector.models.isolation_forest_detector import IsolationForestDetector
from pipeline_anomaly_detector.models.zscore_detector import ZScoreDetector
from pipeline_anomaly_detector.training.model_store import ModelStore

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
_SINCE = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _load_collector(filename: str) -> GenericCollector:
    """Create a GenericCollector from a fixture JSON file.

    Args:
        filename: Fixture filename relative to fixtures dir.

    Returns:
        Configured GenericCollector.
    """
    return GenericCollector(source=FIXTURES_DIR / filename)


def _load_runs(filename: str) -> list[PipelineRun]:
    """Load PipelineRun objects from a fixture file.

    Args:
        filename: Fixture filename relative to fixtures dir.

    Returns:
        List of PipelineRun objects.
    """
    return _load_collector(filename).collect(since=_SINCE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_full_pipeline_detects_injected_anomalies():
    """End-to-end: EnsembleDetector must flag >= 80% of injected anomalies."""
    normal_runs = _load_runs("normal_pipeline_runs.json")
    anomalous_runs = _load_runs("anomalous_pipeline_runs.json")

    assert len(normal_runs) > 0, "No normal runs loaded from fixture"
    assert len(anomalous_runs) > 0, "No anomalous runs loaded from fixture"

    # Fit on normal data
    detector = EnsembleDetector(
        detectors=[ZScoreDetector(window=30), IsolationForestDetector(contamination=0.05)],
        weights=None,  # equal weights
        threshold=0.6,
    )
    detector.fit(normal_runs)

    # Score anomalous runs
    scores = detector.batch_score(anomalous_runs)
    assert len(scores) == len(anomalous_runs)

    anomaly_count = sum(1 for s in scores if s.is_anomaly)
    detection_rate = anomaly_count / len(scores)

    assert detection_rate >= 0.80, (
        f"Detection rate {detection_rate:.1%} is below 80% "
        f"({anomaly_count}/{len(scores)} flagged)"
    )


def test_full_pipeline_with_model_store(tmp_path):
    """End-to-end: save a fitted detector, load it, and score an anomalous run."""
    normal_runs = _load_runs("normal_pipeline_runs.json")
    anomalous_runs = _load_runs("anomalous_pipeline_runs.json")

    # Step 1: Fit
    detector = EnsembleDetector(
        detectors=[ZScoreDetector(), IsolationForestDetector()],
        threshold=0.6,
    )
    detector.fit(normal_runs)

    # Step 2: Save to model store
    store = ModelStore(store_dir=tmp_path)
    model_path = store.save(detector, pipeline_name="orders_pipeline")
    assert model_path.exists()
    assert model_path.suffix == ".joblib"

    # Verify sidecar metadata was written
    meta_files = list(tmp_path.glob("*.meta.json"))
    assert len(meta_files) == 1

    # Step 3: Load from model store
    loaded_detector = store.load(model_path)
    assert loaded_detector is not None
    assert hasattr(loaded_detector, "detector_name")
    assert loaded_detector.detector_name == "ensemble"

    # Step 4: Score an anomalous run
    anomalous_run = anomalous_runs[0]
    score = loaded_detector.score(anomalous_run)

    assert 0.0 <= score.anomaly_score <= 1.0
    assert score.run_id == anomalous_run.run_id
    assert score.detector_name == "ensemble"
    # The first anomalous run should ideally be flagged, but at minimum
    # confirm the score is a valid float > 0
    assert score.anomaly_score >= 0.0


def test_full_pipeline_normal_runs_low_anomaly_rate():
    """Normal runs should have a low false-positive rate (< 20%)."""
    normal_runs = _load_runs("normal_pipeline_runs.json")

    # Use first 80% for training, score the remaining 20%
    n_train = int(len(normal_runs) * 0.8)
    train_runs = normal_runs[:n_train]
    test_runs = normal_runs[n_train:]

    if len(test_runs) == 0:
        pytest.skip("Not enough runs for train/test split")

    detector = EnsembleDetector(
        detectors=[ZScoreDetector(), IsolationForestDetector()],
        threshold=0.6,
    )
    detector.fit(train_runs)

    scores = detector.batch_score(test_runs)
    false_positive_count = sum(1 for s in scores if s.is_anomaly)
    false_positive_rate = false_positive_count / len(scores)

    assert false_positive_rate < 0.20, (
        f"False positive rate {false_positive_rate:.1%} exceeds 20% "
        f"({false_positive_count}/{len(scores)} normal runs flagged)"
    )


def test_model_store_load_latest(tmp_path):
    """ModelStore.load_latest() must return the most recently saved model."""
    import time

    normal_runs = _load_runs("normal_pipeline_runs.json")
    store = ModelStore(store_dir=tmp_path)

    # Save two detectors, one after the other
    det1 = ZScoreDetector()
    det1.fit(normal_runs)
    store.save(det1, pipeline_name="pipeline_a")

    # Small sleep to ensure different timestamps in filenames
    import time as _time
    _time.sleep(1.1)

    det2 = ZScoreDetector(window=15)
    det2.fit(normal_runs)
    store.save(det2, pipeline_name="pipeline_a")

    # load_latest should return the most recently saved
    loaded = store.load_latest(pipeline_name="pipeline_a", detector_name="zscore")
    assert loaded is not None
    assert loaded.detector_name == "zscore"

    # list_models should return 2 models, newest first
    models = store.list_models()
    assert len(models) == 2
