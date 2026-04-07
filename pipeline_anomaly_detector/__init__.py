"""Pipeline Anomaly Detector.

An ML-based anomaly detection system for data pipelines supporting dbt,
Airflow, and generic pipeline run data.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

__version__ = "0.1.0"
__all__ = ["PipelineRun", "__version__"]


class PipelineRun(BaseModel):
    """Represents a single execution of a data pipeline.

    Attributes:
        run_id: Unique identifier for this pipeline run.
        pipeline_name: Name of the pipeline (e.g., dbt model name, DAG task id).
        start_time: UTC timestamp when the run started.
        end_time: UTC timestamp when the run ended.
        duration_seconds: Total wall-clock duration in seconds.
        rows_processed: Number of rows successfully processed.
        rows_rejected: Number of rows that failed validation or processing.
        null_rate: Mapping of column name to fraction of null values (0.0–1.0).
        execution_cost_usd: Optional cloud compute cost in USD.
        status: Terminal status of the run.
        metadata: Arbitrary key-value pairs for additional context.
    """

    run_id: str = Field(..., description="Unique run identifier")
    pipeline_name: str = Field(..., description="Name of the pipeline")
    start_time: datetime = Field(..., description="Run start timestamp (UTC)")
    end_time: datetime = Field(..., description="Run end timestamp (UTC)")
    duration_seconds: float = Field(..., ge=0, description="Wall-clock duration in seconds")
    rows_processed: int = Field(..., ge=0, description="Rows successfully processed")
    rows_rejected: int = Field(default=0, ge=0, description="Rows rejected/failed")
    null_rate: dict[str, float] = Field(
        default_factory=dict,
        description="Column name → null fraction mapping",
    )
    execution_cost_usd: Optional[float] = Field(
        default=None, ge=0, description="Optional compute cost in USD"
    )
    status: Literal["success", "failed", "skipped"] = Field(
        ..., description="Terminal status of the run"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary additional context"
    )

    @model_validator(mode="after")
    def validate_end_after_start(self) -> "PipelineRun":
        """Ensure end_time is not before start_time."""
        if self.end_time < self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be >= start_time ({self.start_time})"
            )
        return self

    model_config = {"frozen": False}
