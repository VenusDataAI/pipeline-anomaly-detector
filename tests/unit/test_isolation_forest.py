"""Unit tests for IsolationForestDetector."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pipeline_anomaly_detector import PipelineRun
from pipeline_anomaly_detector.models.base_detector import AnomalyScore
from pipeline_anomaly_detector.models.isolation_forest_detector import IsolationForestDetector

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _load_runs(filename: str) -> list[PipelineRun]:
    """Load PipelineRun objects from a fixture file.

    Args:
        filename: Fixture filename relative to fixtures dir.

    Returns:
        List of PipelineRun objects.
    """
    path = FIXTURES_DIR / filename
    with path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    return [PipelineRun.model_validate(r) for r in records]


@pytest.fixture
def normal_runs() -> list[PipelineRun]:
    """Normal pipeline runs fixture."""
    return _load_runs("normal_pipeline_runs.json")


@pytest.fixture
def anomalous_runs() -> list[PipelineRun]:
    """Anomalous pipeline runs fixture."""
    return _load_runs("anomalous_pipeline_runs.json")


@pytest.fixture
def fitted_detector(normal_runs) -> IsolationForestDetector:
    """Return an IsolationForestDetector fitted on normal runs."""
    det = IsolationForestDetector(contamination=0.05, threshold=0.5)
    det.fit(normal_runs)
    return det


def test_fit_on_normal_data(normal_runs):
    """fit() must complete without raising on normal data."""
    det = IsolationForestDetector()
    det.fit(normal_runs)  # Should not raise
    assert det._is_fitted is True


def test_score_returns_anomaly_score(fitted_detector, normal_runs):
    """score() must return an AnomalyScore with correct fields."""
    run = normal_runs[0]
    result = fitted_detector.score(run)

    assert isinstance(result, AnomalyScore)
    assert result.run_id == run.run_id
    assert result.pipeline_name == run.pipeline_name
    assert result.detector_name == "isolation_forest"
    assert isinstance(result.anomaly_score, float)
    assert isinstance(result.is_anomaly, bool)
    assert isinstance(result.contributing_features, list)
    assert isinstance(result.timestamp, datetime)


def test_anomaly_detection_rate_above_80pct(normal_runs, anomalous_runs):
    """At least 80% of injected anomalous runs must be flagged as is_anomaly."""
    det = IsolationForestDetector(contamination=0.05, threshold=0.5)
    det.fit(normal_runs)

    scores = det.batch_score(anomalous_runs)
    anomaly_count = sum(1 for s in scores if s.is_anomaly)
    detection_rate = anomaly_count / len(scores)

    assert detection_rate >= 0.80, (
        f"Detection rate {detection_rate:.1%} is below 80% "
        f"({anomaly_count}/{len(scores)} anomalies flagged)"
    )


def test_contributing_features_top3(fitted_detector, normal_runs):
    """contributing_features must have at most 3 items."""
    run = normal_runs[0]
    result = fitted_detector.score(run)
    assert len(result.contributing_features) <= 3


def test_anomaly_score_in_range(fitted_detector, normal_runs):
    """anomaly_score must be in [0.0, 1.0] for all scored runs."""
    scores = fitted_detector.batch_score(normal_runs[:10])
    for s in scores:
        assert 0.0 <= s.anomaly_score <= 1.0, (
            f"Score {s.anomaly_score} for run {s.run_id} is outside [0, 1]"
        )
