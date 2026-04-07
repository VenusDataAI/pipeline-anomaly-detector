"""Trainer orchestrates collection and fitting of detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict

import structlog

from pipeline_anomaly_detector import PipelineRun
from pipeline_anomaly_detector.collectors.base_collector import BaseCollector
from pipeline_anomaly_detector.models.base_detector import BaseDetector

log = structlog.get_logger(__name__)


@dataclass
class FitResult:
    """Result of a :class:`Trainer` fit operation.

    Attributes:
        detector: The fitted detector instance.
        training_stats: Summary statistics about the training run:
            ``n_runs`` (int), ``n_pipelines`` (int), and
            ``feature_summary`` (dict mapping feature name -> basic stats).
        fitted_at: UTC timestamp when fitting completed.
    """

    detector: BaseDetector
    training_stats: dict
    fitted_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )


class Trainer:
    """High-level coordinator for collecting data and fitting detectors.

    Example::

        trainer = Trainer()
        result = trainer.fit(
            collector=GenericCollector("runs.jsonl"),
            since=datetime(2024, 1, 1, tzinfo=timezone.utc),
            detector=EnsembleDetector([ZScoreDetector(), IsolationForestDetector()]),
        )
        print(result.training_stats)
    """

    def fit(
        self,
        collector: BaseCollector,
        since: datetime,
        detector: BaseDetector,
        pipeline_names: list[str] | None = None,
    ) -> FitResult:
        """Collect runs and fit the detector.

        Args:
            collector: A :class:`~pipeline_anomaly_detector.collectors.BaseCollector`
                used to fetch pipeline runs.
            since: Lower bound timestamp for run collection.
            detector: An unfitted :class:`~pipeline_anomaly_detector.models.BaseDetector`
                to train.
            pipeline_names: Optional list of pipeline names to include.
                If ``None``, all pipelines are used.

        Returns:
            A :class:`FitResult` containing the fitted detector and training
            statistics.
        """
        log.info(
            "trainer_collecting",
            collector=type(collector).__name__,
            since=since.isoformat(),
            detector=detector.detector_name,
            pipeline_names=pipeline_names,
        )

        runs = collector.collect(since=since)

        if pipeline_names is not None:
            pipeline_set = set(pipeline_names)
            runs = [r for r in runs if r.pipeline_name in pipeline_set]
            log.debug(
                "trainer_filtered_by_pipeline",
                n_runs=len(runs),
                pipeline_names=pipeline_names,
            )

        log.info(
            "trainer_fitting",
            n_runs=len(runs),
            detector=detector.detector_name,
        )
        detector.fit(runs)

        stats = self._compute_stats(runs)

        log.info(
            "trainer_fit_complete",
            n_runs=stats["n_runs"],
            n_pipelines=stats["n_pipelines"],
            detector=detector.detector_name,
        )

        return FitResult(
            detector=detector,
            training_stats=stats,
            fitted_at=datetime.now(tz=timezone.utc),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stats(runs: list[PipelineRun]) -> dict:
        """Compute summary statistics for the training set.

        Args:
            runs: Training pipeline runs.

        Returns:
            Dict with keys ``n_runs``, ``n_pipelines``, ``feature_summary``.
        """
        if not runs:
            return {
                "n_runs": 0,
                "n_pipelines": 0,
                "feature_summary": {},
            }

        per_pipeline: dict[str, list[PipelineRun]] = defaultdict(list)
        for run in runs:
            per_pipeline[run.pipeline_name].append(run)

        # Duration stats
        durations = [r.duration_seconds for r in runs]
        rows = [r.rows_processed for r in runs]

        feature_summary = {
            "duration_seconds": {
                "mean": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
            },
            "rows_processed": {
                "mean": sum(rows) / len(rows),
                "min": min(rows),
                "max": max(rows),
            },
            "status_counts": {
                status: sum(1 for r in runs if r.status == status)
                for status in ("success", "failed", "skipped")
            },
        }

        return {
            "n_runs": len(runs),
            "n_pipelines": len(per_pipeline),
            "feature_summary": feature_summary,
        }
