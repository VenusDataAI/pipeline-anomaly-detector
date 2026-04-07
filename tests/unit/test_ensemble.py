"""Unit tests for EnsembleDetector."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pipeline_anomaly_detector import PipelineRun
from pipeline_anomaly_detector.models.base_detector import AnomalyScore, BaseDetector
from pipeline_anomaly_detector.models.ensemble_detector import EnsembleDetector

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Helper: mock detector factory
# ---------------------------------------------------------------------------

def _make_mock_detector(
    name: str,
    score_value: float,
    contributing: list[str],
) -> MagicMock:
    """Create a mock BaseDetector with a fixed score output.

    Args:
        name: detector_name property value.
        score_value: Fixed anomaly_score to return.
        contributing: Fixed contributing_features to return.

    Returns:
        MagicMock instance configured as a BaseDetector.
    """
    mock = MagicMock(spec=BaseDetector)
    mock.detector_name = name
    mock.threshold = 0.5

    def _score(run: PipelineRun) -> AnomalyScore:
        return AnomalyScore(
            run_id=run.run_id,
            pipeline_name=run.pipeline_name,
            anomaly_score=score_value,
            is_anomaly=score_value >= 0.5,
            contributing_features=contributing,
            detector_name=name,
            timestamp=datetime.now(tz=timezone.utc),
        )

    mock.score.side_effect = _score
    mock.batch_score.side_effect = lambda runs: [_score(r) for r in runs]
    mock.fit.return_value = mock
    return mock


def _dummy_run(
    run_id: str = "test_run",
    pipeline_name: str = "test_pipeline",
    anomaly_score: float = 0.0,
) -> PipelineRun:
    """Create a minimal PipelineRun for testing.

    Args:
        run_id: Run identifier.
        pipeline_name: Pipeline name.
        anomaly_score: Unused — present for API symmetry.

    Returns:
        PipelineRun instance.
    """
    return PipelineRun(
        run_id=run_id,
        pipeline_name=pipeline_name,
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
        duration_seconds=3600.0,
        rows_processed=1000,
        rows_rejected=0,
        null_rate={},
        status="success",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ensemble_score_is_weighted_average():
    """Ensemble anomaly_score must equal the weighted average of sub-scores."""
    det_a = _make_mock_detector("a", score_value=0.8, contributing=["feat_a"])
    det_b = _make_mock_detector("b", score_value=0.2, contributing=["feat_b"])

    ensemble = EnsembleDetector(
        detectors=[det_a, det_b],
        weights=[0.6, 0.4],
        threshold=0.6,
    )
    run = _dummy_run()
    result = ensemble.score(run)

    expected = 0.6 * 0.8 + 0.4 * 0.2  # = 0.56
    assert result.anomaly_score == pytest.approx(expected, abs=1e-6)


def test_equal_weights_default():
    """When weights=None, equal weights must be applied."""
    det_a = _make_mock_detector("a", score_value=1.0, contributing=[])
    det_b = _make_mock_detector("b", score_value=0.0, contributing=[])

    ensemble = EnsembleDetector(detectors=[det_a, det_b], weights=None)
    run = _dummy_run()
    result = ensemble.score(run)

    # Equal weights: (1.0 + 0.0) / 2 = 0.5
    assert result.anomaly_score == pytest.approx(0.5, abs=1e-6)
    assert ensemble._weights == pytest.approx([0.5, 0.5], abs=1e-6)


def test_mismatched_weights_normalized():
    """Provided weights must be normalised to sum to 1.0."""
    det_a = _make_mock_detector("a", score_value=1.0, contributing=[])
    det_b = _make_mock_detector("b", score_value=0.0, contributing=[])

    # weights=[0.7, 0.3] already sum to 1.0, but test that normalisation works
    ensemble = EnsembleDetector(
        detectors=[det_a, det_b],
        weights=[0.7, 0.3],
    )
    assert sum(ensemble._weights) == pytest.approx(1.0, abs=1e-9)

    run = _dummy_run()
    result = ensemble.score(run)
    # 0.7 * 1.0 + 0.3 * 0.0 = 0.7
    assert result.anomaly_score == pytest.approx(0.7, abs=1e-6)


def test_contributing_features_union():
    """contributing_features must include features from all sub-detectors (deduplicated)."""
    det_a = _make_mock_detector("a", score_value=0.8, contributing=["feat_a", "shared"])
    det_b = _make_mock_detector("b", score_value=0.8, contributing=["feat_b", "shared"])

    ensemble = EnsembleDetector(detectors=[det_a, det_b])
    run = _dummy_run()
    result = ensemble.score(run)

    # All unique features should be present
    assert "feat_a" in result.contributing_features
    assert "feat_b" in result.contributing_features
    assert "shared" in result.contributing_features
    # "shared" must appear only once
    assert result.contributing_features.count("shared") == 1


def test_is_anomaly_above_threshold():
    """is_anomaly must be True when weighted score exceeds threshold."""
    det_a = _make_mock_detector("a", score_value=0.9, contributing=[])
    det_b = _make_mock_detector("b", score_value=0.9, contributing=[])

    ensemble = EnsembleDetector(
        detectors=[det_a, det_b],
        threshold=0.6,
    )
    run = _dummy_run()
    result = ensemble.score(run)

    assert result.anomaly_score > 0.6
    assert result.is_anomaly is True


def test_is_anomaly_below_threshold():
    """is_anomaly must be False when weighted score is below threshold."""
    det_a = _make_mock_detector("a", score_value=0.1, contributing=[])
    det_b = _make_mock_detector("b", score_value=0.1, contributing=[])

    ensemble = EnsembleDetector(
        detectors=[det_a, det_b],
        threshold=0.6,
    )
    run = _dummy_run()
    result = ensemble.score(run)

    assert result.is_anomaly is False


def test_fit_calls_all_subdetectors():
    """fit() must call fit() on every sub-detector."""
    det_a = _make_mock_detector("a", score_value=0.0, contributing=[])
    det_b = _make_mock_detector("b", score_value=0.0, contributing=[])

    ensemble = EnsembleDetector(detectors=[det_a, det_b])

    runs = [_dummy_run(run_id=f"r{i}") for i in range(5)]
    ensemble.fit(runs)

    det_a.fit.assert_called_once_with(runs)
    det_b.fit.assert_called_once_with(runs)
