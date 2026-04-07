"""Abstract base class for pipeline run collectors."""

from __future__ import annotations

import abc
from datetime import datetime

import structlog

from pipeline_anomaly_detector import PipelineRun

log = structlog.get_logger(__name__)


class BaseCollector(abc.ABC):
    """Abstract base class for all pipeline run collectors.

    Subclasses must implement :meth:`collect` to return a list of
    :class:`~pipeline_anomaly_detector.PipelineRun` objects representing
    pipeline executions since a given timestamp.
    """

    @abc.abstractmethod
    def collect(self, since: datetime) -> list[PipelineRun]:
        """Collect pipeline runs that started at or after *since*.

        Args:
            since: Lower-bound timestamp (inclusive). Only runs with
                ``start_time >= since`` are returned.

        Returns:
            A list of :class:`~pipeline_anomaly_detector.PipelineRun`
            objects ordered by ``start_time`` ascending.
        """
        ...

    def collect_all(self) -> list[PipelineRun]:
        """Collect all available pipeline runs regardless of start time.

        Returns:
            A list of all :class:`~pipeline_anomaly_detector.PipelineRun`
            objects available to this collector.
        """
        from datetime import timezone

        return self.collect(since=datetime.min.replace(tzinfo=timezone.utc))
