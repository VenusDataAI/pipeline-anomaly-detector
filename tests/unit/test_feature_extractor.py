"""Unit tests for the FeatureExtractor."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from pipeline_anomaly_detector import PipelineRun
from pipeline_anomaly_detector.features.feature_extractor import FeatureExtractor
from pipeline_anomaly_detector.features.feature_registry import FEATURE_REGISTRY

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _load_runs(filename: str) -> list[PipelineRun]:
    """Load PipelineRun objects from a fixture JSON file.

    Args:
        filename: Fixture file name (relative to fixtures dir).

    Returns:
        List of validated PipelineRun objects.
    """
    path = FIXTURES_DIR / filename
    with path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    return [PipelineRun.model_validate(r) for r in records]


@pytest.fixture
def normal_runs() -> list[PipelineRun]:
    """Return all normal pipeline runs from fixtures."""
    return _load_runs("normal_pipeline_runs.json")


@pytest.fixture
def single_run(normal_runs) -> PipelineRun:
    """Return the first normal pipeline run."""
    return normal_runs[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_11_features_present(normal_runs):
    """All 11 registered feature names must appear as DataFrame columns."""
    extractor = FeatureExtractor()
    df = extractor.fit_transform(normal_runs)
    expected = set(FEATURE_REGISTRY.feature_names)
    actual = set(df.columns)
    assert expected == actual, f"Missing: {expected - actual}, Extra: {actual - expected}"


def test_duration_seconds_raw(single_run):
    """duration_seconds must exactly match PipelineRun.duration_seconds."""
    extractor = FeatureExtractor()
    df = extractor.fit_transform([single_run])
    assert df.loc[single_run.run_id, "duration_seconds"] == pytest.approx(
        single_run.duration_seconds, rel=1e-6
    )


def test_log1p_transformation(single_run):
    """rows_processed_log1p must equal log1p(rows_processed)."""
    extractor = FeatureExtractor()
    df = extractor.fit_transform([single_run])
    expected = math.log1p(single_run.rows_processed)
    actual = float(df.loc[single_run.run_id, "rows_processed_log1p"])
    assert actual == pytest.approx(expected, rel=1e-6)


def test_rejection_rate_zero_denom():
    """rejection_rate must be 0.0 when rows_processed == 0."""
    run = PipelineRun(
        run_id="test_zero_denom",
        pipeline_name="test_pipeline",
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
        duration_seconds=3600.0,
        rows_processed=0,
        rows_rejected=0,
        null_rate={},
        status="success",
    )
    extractor = FeatureExtractor()
    df = extractor.fit_transform([run])
    assert float(df.loc["test_zero_denom", "rejection_rate"]) == 0.0


def test_windowed_features_less_than_30_runs():
    """With fewer than 30 historical runs, duration_z should be 0.0 (graceful)."""
    # Build 5 runs for the same pipeline
    runs = []
    for i in range(5):
        runs.append(
            PipelineRun(
                run_id=f"run_{i:03d}",
                pipeline_name="small_pipeline",
                start_time=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
                end_time=datetime(2024, 1, i + 1, 1, tzinfo=timezone.utc),
                duration_seconds=float(3600 + i * 10),
                rows_processed=1000,
                rows_rejected=0,
                null_rate={},
                status="success",
            )
        )
    extractor = FeatureExtractor()
    # Should NOT raise
    df = extractor.fit_transform(runs)
    assert "duration_z" in df.columns
    # Values should be numeric (0.0 or valid float), not NaN for all
    duration_z_vals = df["duration_z"].values
    for val in duration_z_vals:
        assert not math.isnan(float(val)) or float(val) == 0.0, (
            f"duration_z should not be NaN, got {val}"
        )


def test_is_weekend():
    """is_weekend must be 1 for Saturday/Sunday and 0 for weekdays."""
    # 2024-01-06 is a Saturday
    saturday_run = PipelineRun(
        run_id="sat_run",
        pipeline_name="test",
        start_time=datetime(2024, 1, 6, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 6, 11, 0, tzinfo=timezone.utc),
        duration_seconds=3600.0,
        rows_processed=1000,
        rows_rejected=0,
        null_rate={},
        status="success",
    )
    # 2024-01-07 is a Sunday
    sunday_run = PipelineRun(
        run_id="sun_run",
        pipeline_name="test",
        start_time=datetime(2024, 1, 7, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 7, 11, 0, tzinfo=timezone.utc),
        duration_seconds=3600.0,
        rows_processed=1000,
        rows_rejected=0,
        null_rate={},
        status="success",
    )
    # 2024-01-08 is a Monday
    monday_run = PipelineRun(
        run_id="mon_run",
        pipeline_name="test",
        start_time=datetime(2024, 1, 8, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 8, 11, 0, tzinfo=timezone.utc),
        duration_seconds=3600.0,
        rows_processed=1000,
        rows_rejected=0,
        null_rate={},
        status="success",
    )
    extractor = FeatureExtractor()
    df = extractor.fit_transform([saturday_run, sunday_run, monday_run])
    assert int(df.loc["sat_run", "is_weekend"]) == 1
    assert int(df.loc["sun_run", "is_weekend"]) == 1
    assert int(df.loc["mon_run", "is_weekend"]) == 0


def test_status_is_success():
    """status_is_success must be 1 for 'success' and 0 for other statuses."""
    success_run = PipelineRun(
        run_id="success_run",
        pipeline_name="test",
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
        duration_seconds=3600.0,
        rows_processed=1000,
        rows_rejected=0,
        null_rate={},
        status="success",
    )
    failed_run = PipelineRun(
        run_id="failed_run",
        pipeline_name="test",
        start_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 2, 1, tzinfo=timezone.utc),
        duration_seconds=3600.0,
        rows_processed=0,
        rows_rejected=1000,
        null_rate={},
        status="failed",
    )
    extractor = FeatureExtractor()
    df = extractor.fit_transform([success_run, failed_run])
    assert int(df.loc["success_run", "status_is_success"]) == 1
    assert int(df.loc["failed_run", "status_is_success"]) == 0


def test_null_rate_max():
    """null_rate_max must equal the maximum value across all columns."""
    run = PipelineRun(
        run_id="null_run",
        pipeline_name="test",
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
        duration_seconds=3600.0,
        rows_processed=1000,
        rows_rejected=0,
        null_rate={"col_a": 0.05, "col_b": 0.15, "col_c": 0.02},
        status="success",
    )
    extractor = FeatureExtractor()
    df = extractor.fit_transform([run])
    assert float(df.loc["null_run", "null_rate_max"]) == pytest.approx(0.15, rel=1e-6)


def test_hour_of_day():
    """hour_of_day must correctly extract the UTC hour from start_time."""
    run = PipelineRun(
        run_id="hour_run",
        pipeline_name="test",
        start_time=datetime(2024, 1, 1, 14, 30, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 1, 15, 30, tzinfo=timezone.utc),
        duration_seconds=3600.0,
        rows_processed=1000,
        rows_rejected=0,
        null_rate={},
        status="success",
    )
    extractor = FeatureExtractor()
    df = extractor.fit_transform([run])
    assert int(df.loc["hour_run", "hour_of_day"]) == 14


def test_fit_transform_returns_dataframe(normal_runs):
    """fit_transform must return a pandas DataFrame instance."""
    extractor = FeatureExtractor()
    df = extractor.fit_transform(normal_runs)
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "run_id"
    assert len(df) == len(normal_runs)
