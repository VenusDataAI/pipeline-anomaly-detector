"""Unit tests for ZScoreDetector."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pipeline_anomaly_detector import PipelineRun
from pipeline_anomaly_detector.models.base_detector import AnomalyScore
from pipeline_anomaly_detector.models.zscore_detector import ZScoreDetector

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
def fitted_detector(normal_runs) -> ZScoreDetector:
    """Return a ZScoreDetector fitted on normal runs."""
    det = ZScoreDetector(window=30, threshold=0.5)
    det.fit(normal_runs)
    return det


def test_fit_on_normal_data(normal_runs):
    """fit() must complete without raising on normal training data."""
    det = ZScoreDetector()
    det.fit(normal_runs)
    assert det._is_fitted is True


def test_score_returns_anomaly_score(fitted_detector, normal_runs):
    """score() must return a well-formed AnomalyScore."""
    run = normal_runs[0]
    result = fitted_detector.score(run)

    assert isinstance(result, AnomalyScore)
    assert result.run_id == run.run_id
    assert result.pipeline_name == run.pipeline_name
    assert result.detector_name == "zscore"
    assert 0.0 <= result.anomaly_score <= 1.0
    assert isinstance(result.is_anomaly, bool)
    assert isinstance(result.contributing_features, list)
    assert len(result.contributing_features) <= 3


def test_anomaly_detection_rate_above_80pct(normal_runs, anomalous_runs):
    """At least 80% of anomalous runs must be flagged."""
    det = ZScoreDetector(window=30, threshold=0.5)
    det.fit(normal_runs)

    scores = det.batch_score(anomalous_runs)
    anomaly_count = sum(1 for s in scores if s.is_anomaly)
    detection_rate = anomaly_count / len(scores)

    assert detection_rate >= 0.80, (
        f"Detection rate {detection_rate:.1%} is below 80% "
        f"({anomaly_count}/{len(scores)} anomalies flagged)"
    )


def test_contributing_features_match_spiked_feature(normal_runs):
    """For a severe duration spike, duration-related feature must be in contributing_features."""
    det = ZScoreDetector(window=30, threshold=0.5)
    det.fit(normal_runs)

    # Create a run with an extreme duration spike (10x normal ~3600s)
    extreme_run = PipelineRun(
        run_id="extreme_duration_run",
        pipeline_name="orders_pipeline",  # must match fixture pipeline name
        start_time=datetime(2024, 7, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 7, 1, 10, tzinfo=timezone.utc),
        duration_seconds=36000.0,  # 10 hours — very anomalous
        rows_processed=100000,
        rows_rejected=0,
        null_rate={"col_a": 0.01, "col_b": 0.01, "col_c": 0.01},
        status="success",
    )

    result = det.score(extreme_run)

    # At least one duration-related feature should appear
    duration_features = {"duration_seconds", "duration_z"}
    assert any(f in duration_features for f in result.contributing_features), (
        f"Expected a duration feature in contributing_features, "
        f"got: {result.contributing_features}"
    )


def test_anomaly_score_capped_at_1():
    """anomaly_score must never exceed 1.0, even for extreme z-scores."""
    # Fit on a very tight distribution
    base_runs = [
        PipelineRun(
            run_id=f"base_{i}",
            pipeline_name="tight_pipeline",
            start_time=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, i + 1, 1, tzinfo=timezone.utc),
            duration_seconds=100.0,  # very consistent
            rows_processed=1000,
            rows_rejected=0,
            null_rate={},
            status="success",
        )
        for i in range(30)
    ]

    det = ZScoreDetector(window=30, threshold=0.5)
    det.fit(base_runs)

    # Score a run that is 1000x the normal duration
    extreme_run = PipelineRun(
        run_id="extreme_run",
        pipeline_name="tight_pipeline",
        start_time=datetime(2024, 3, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 3, 2, tzinfo=timezone.utc),
        duration_seconds=100_000.0,
        rows_processed=1000,
        rows_rejected=0,
        null_rate={},
        status="success",
    )

    result = det.score(extreme_run)
    assert result.anomaly_score <= 1.0, (
        f"anomaly_score {result.anomaly_score} exceeds 1.0"
    )
