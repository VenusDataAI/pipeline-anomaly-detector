"""Unit tests for the Scorer class."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline_anomaly_detector import PipelineRun
from pipeline_anomaly_detector.models.base_detector import AnomalyScore, BaseDetector
from pipeline_anomaly_detector.scoring.alert_router import AlertRouter
from pipeline_anomaly_detector.scoring.scorer import Scorer

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_run(
    run_id: str = "test_run",
    pipeline_name: str = "test_pipeline",
) -> PipelineRun:
    """Create a minimal PipelineRun for testing.

    Args:
        run_id: Run identifier.
        pipeline_name: Pipeline name.

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


def _make_mock_detector(is_anomaly: bool = False, score: float = 0.1) -> MagicMock:
    """Build a mock BaseDetector with a fixed output.

    Args:
        is_anomaly: Whether to flag the run as anomalous.
        score: Anomaly score to return.

    Returns:
        MagicMock detector.
    """
    mock = MagicMock(spec=BaseDetector)
    mock.detector_name = "mock_detector"
    mock.threshold = 0.5

    def _score(run: PipelineRun) -> AnomalyScore:
        return AnomalyScore(
            run_id=run.run_id,
            pipeline_name=run.pipeline_name,
            anomaly_score=score,
            is_anomaly=is_anomaly,
            contributing_features=["duration_seconds"] if is_anomaly else [],
            detector_name="mock_detector",
            timestamp=datetime.now(tz=timezone.utc),
        )

    mock.score.side_effect = _score
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_score_persists_to_sqlite(tmp_path):
    """Scoring a run must insert a row into the SQLite database."""
    db_path = tmp_path / "scores.db"
    mock_det = _make_mock_detector(is_anomaly=False, score=0.1)
    scorer = Scorer(detector=mock_det, db_path=db_path)

    run = _dummy_run(run_id="persist_test_run")
    scorer.score(run)

    history = scorer.get_history(pipeline_name="test_pipeline", limit=10)
    run_ids = [h["run_id"] for h in history]
    assert "persist_test_run" in run_ids


def test_score_batch_returns_list(tmp_path):
    """score_batch must return a list with the same length as the input."""
    db_path = tmp_path / "scores.db"
    mock_det = _make_mock_detector(is_anomaly=False, score=0.1)
    scorer = Scorer(detector=mock_det, db_path=db_path)

    runs = [_dummy_run(run_id=f"run_{i}") for i in range(5)]
    scores = scorer.score_batch(runs)

    assert isinstance(scores, list)
    assert len(scores) == 5


def test_alert_router_called_on_anomaly(tmp_path):
    """AlertRouter.route() must be called when a run is flagged as anomalous."""
    db_path = tmp_path / "scores.db"
    mock_det = _make_mock_detector(is_anomaly=True, score=0.9)
    mock_router = MagicMock(spec=AlertRouter)

    scorer = Scorer(detector=mock_det, db_path=db_path, alert_router=mock_router)
    run = _dummy_run(run_id="anomaly_run")
    scorer.score(run)

    mock_router.route.assert_called_once()
    call_args = mock_router.route.call_args[0][0]
    assert call_args.run_id == "anomaly_run"
    assert call_args.is_anomaly is True


def test_alert_router_not_called_on_normal(tmp_path):
    """AlertRouter.route() must NOT be called when a run is normal."""
    db_path = tmp_path / "scores.db"
    mock_det = _make_mock_detector(is_anomaly=False, score=0.1)
    mock_router = MagicMock(spec=AlertRouter)

    scorer = Scorer(detector=mock_det, db_path=db_path, alert_router=mock_router)
    run = _dummy_run(run_id="normal_run")
    scorer.score(run)

    mock_router.route.assert_not_called()


def test_deduplication_skips_repeat_alert(tmp_path):
    """The second alert for the same pipeline in the same window must be suppressed."""
    db_path = tmp_path / "scores.db"

    # Real AlertRouter (not a mock) so dedup logic is exercised
    router = AlertRouter(
        config={"test_pipeline": [{"type": "log_only"}]},
        dedup_window_minutes=60,
    )

    mock_det = _make_mock_detector(is_anomaly=True, score=0.9)
    scorer = Scorer(detector=mock_det, db_path=db_path, alert_router=router)

    # Score the same pipeline twice
    run1 = _dummy_run(run_id="dup_run_1", pipeline_name="test_pipeline")
    run2 = _dummy_run(run_id="dup_run_2", pipeline_name="test_pipeline")

    scorer.score(run1)
    scorer.score(run2)

    # After the first alert the bucket is marked; second call should be a no-op.
    # The sent set should contain exactly one entry for the pipeline.
    sent_pipelines = {key[0] for key in router._sent}
    assert "test_pipeline" in sent_pipelines
    # Only 1 unique (pipeline, bucket) pair was added
    assert len(router._sent) == 1


def test_get_history_returns_list(tmp_path):
    """get_history() must return a list of dicts with the expected keys."""
    db_path = tmp_path / "scores.db"
    mock_det = _make_mock_detector(is_anomaly=False, score=0.2)
    scorer = Scorer(detector=mock_det, db_path=db_path)

    runs = [_dummy_run(run_id=f"hist_run_{i}") for i in range(3)]
    scorer.score_batch(runs)

    history = scorer.get_history("test_pipeline", limit=10)
    assert isinstance(history, list)
    assert len(history) == 3

    required_keys = {"run_id", "pipeline_name", "anomaly_score", "is_anomaly",
                     "contributing_features", "detector_name", "timestamp"}
    for entry in history:
        assert required_keys.issubset(set(entry.keys()))
