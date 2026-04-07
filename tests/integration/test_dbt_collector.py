"""Integration tests for DbtCollector."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pipeline_anomaly_detector import PipelineRun
from pipeline_anomaly_detector.collectors.dbt_collector import DbtCollector


# ---------------------------------------------------------------------------
# Fixture data builders
# ---------------------------------------------------------------------------

def _make_run_results(results: list[dict]) -> dict:
    """Wrap a list of results in a dbt run_results.json structure.

    Args:
        results: List of result dicts.

    Returns:
        Dict matching the dbt run_results.json schema.
    """
    return {
        "metadata": {
            "dbt_schema_version": "https://schemas.getdbt.com/dbt/run-results/v4.json",
            "dbt_version": "1.7.0",
            "generated_at": "2024-01-15T10:00:00.000000Z",
            "invocation_id": "test-invocation-123",
        },
        "results": results,
        "elapsed_time": 42.5,
    }


def _make_result(
    unique_id: str = "model.my_project.orders",
    status: str = "success",
    duration_seconds: float = 15.3,
    started_at: str = "2024-01-15T09:00:00.000000Z",
    include_adapter_response: bool = True,
    rows_affected: int = 5000,
) -> dict:
    """Build a single dbt result node dict.

    Args:
        unique_id: dbt unique identifier for the model.
        status: Execution status string.
        duration_seconds: Duration in seconds.
        started_at: ISO timestamp for run start.
        include_adapter_response: Whether to include adapter_response.
        rows_affected: Row count in adapter_response.

    Returns:
        Dict representing a single dbt result node.
    """
    from datetime import timedelta

    # Compute completed_at from started_at + duration
    start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    end_dt = start_dt + timedelta(seconds=duration_seconds)
    completed_at = end_dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

    result: dict = {
        "unique_id": unique_id,
        "status": status,
        "execution_time": duration_seconds,
        "timing": [
            {
                "name": "compile",
                "started_at": started_at,
                "completed_at": started_at,
            },
            {
                "name": "execute",
                "started_at": started_at,
                "completed_at": completed_at,
            },
        ],
    }
    if include_adapter_response:
        result["adapter_response"] = {
            "rows_affected": rows_affected,
            "_message": f"SELECT {rows_affected}",
        }
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_dbt_run_results(tmp_path):
    """DbtCollector must parse a real run_results.json file correctly."""
    results_data = _make_run_results(
        [
            _make_result(
                unique_id="model.project.orders",
                status="success",
                duration_seconds=15.3,
                rows_affected=5000,
            )
        ]
    )

    run_results_file = tmp_path / "run_results.json"
    run_results_file.write_text(json.dumps(results_data), encoding="utf-8")

    collector = DbtCollector(directory_glob=str(run_results_file))
    runs = collector.collect(since=datetime(2024, 1, 1, tzinfo=timezone.utc))

    assert len(runs) == 1
    run = runs[0]

    assert isinstance(run, PipelineRun)
    assert run.pipeline_name == "orders"
    assert run.duration_seconds == pytest.approx(15.3, rel=0.1)
    assert run.rows_processed == 5000
    assert run.status == "success"
    assert run.run_id.startswith("dbt_")
    assert "model.project.orders" in run.run_id or "orders" in run.pipeline_name


def test_missing_adapter_response(tmp_path):
    """DbtCollector must handle runs with no adapter_response gracefully."""
    results_data = _make_run_results(
        [
            _make_result(
                unique_id="model.project.customers",
                status="success",
                duration_seconds=8.0,
                include_adapter_response=False,
            )
        ]
    )

    run_results_file = tmp_path / "run_results.json"
    run_results_file.write_text(json.dumps(results_data), encoding="utf-8")

    collector = DbtCollector(directory_glob=str(run_results_file))
    runs = collector.collect(since=datetime(2024, 1, 1, tzinfo=timezone.utc))

    assert len(runs) == 1
    run = runs[0]
    # rows_processed should default to 0 when adapter_response is absent
    assert run.rows_processed == 0
    # Should still parse correctly
    assert run.pipeline_name == "customers"
    assert run.status == "success"


def test_since_filter(tmp_path):
    """DbtCollector must only return runs at or after the `since` timestamp."""
    # Two results: one early (2024-01-10), one late (2024-02-01)
    results_data = _make_run_results(
        [
            _make_result(
                unique_id="model.project.early_model",
                status="success",
                duration_seconds=5.0,
                started_at="2024-01-10T08:00:00.000000Z",
            ),
            _make_result(
                unique_id="model.project.late_model",
                status="success",
                duration_seconds=7.0,
                started_at="2024-02-01T08:00:00.000000Z",
            ),
        ]
    )

    run_results_file = tmp_path / "run_results.json"
    run_results_file.write_text(json.dumps(results_data), encoding="utf-8")

    collector = DbtCollector(directory_glob=str(run_results_file))

    # Filter: only runs after 2024-01-15
    since_dt = datetime(2024, 1, 15, tzinfo=timezone.utc)
    runs = collector.collect(since=since_dt)

    assert len(runs) == 1
    assert runs[0].pipeline_name == "late_model"


def test_multiple_result_files(tmp_path):
    """DbtCollector must aggregate results from multiple matching files."""
    for i in range(3):
        results_data = _make_run_results(
            [
                _make_result(
                    unique_id=f"model.project.model_{i}",
                    status="success",
                    duration_seconds=float(i + 1),
                    started_at="2024-01-15T09:00:00.000000Z",
                )
            ]
        )
        sub_dir = tmp_path / f"project_{i}" / "target"
        sub_dir.mkdir(parents=True)
        (sub_dir / "run_results.json").write_text(
            json.dumps(results_data), encoding="utf-8"
        )

    collector = DbtCollector(directory_glob=str(tmp_path / "**/run_results.json"))
    runs = collector.collect(since=datetime(2024, 1, 1, tzinfo=timezone.utc))

    assert len(runs) == 3


def test_failed_status_mapped_correctly(tmp_path):
    """DbtCollector must map dbt 'error' status to PipelineRun 'failed'."""
    results_data = _make_run_results(
        [
            _make_result(
                unique_id="model.project.broken_model",
                status="error",
                duration_seconds=2.0,
            )
        ]
    )

    run_results_file = tmp_path / "run_results.json"
    run_results_file.write_text(json.dumps(results_data), encoding="utf-8")

    collector = DbtCollector(directory_glob=str(run_results_file))
    runs = collector.collect(since=datetime(2024, 1, 1, tzinfo=timezone.utc))

    assert len(runs) == 1
    assert runs[0].status == "failed"
