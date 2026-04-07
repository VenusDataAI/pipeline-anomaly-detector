"""Abstract base detector and AnomalyScore dataclass."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from pipeline_anomaly_detector import PipelineRun

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)


@dataclass
class AnomalyScore:
    """Result of scoring a single pipeline run.

    Attributes:
        run_id: The unique identifier of the scored run.
        pipeline_name: The name of the pipeline.
        anomaly_score: Normalised anomaly score in [0.0, 1.0].
            Higher values indicate greater anomalousness.
        is_anomaly: ``True`` when ``anomaly_score >= detector.threshold``.
        contributing_features: Top (up to 3) feature names that contributed
            most to the anomaly score.
        detector_name: Name of the detector that produced this score.
        timestamp: UTC timestamp when the score was computed.
    """

    run_id: str
    pipeline_name: str
    anomaly_score: float
    is_anomaly: bool
    contributing_features: list[str] = field(default_factory=list)
    detector_name: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    def __post_init__(self) -> None:
        """Clamp anomaly_score to [0.0, 1.0]."""
        self.anomaly_score = max(0.0, min(1.0, self.anomaly_score))


class BaseDetector(abc.ABC):
    """Abstract base class for all anomaly detectors.

    Subclasses must implement :meth:`fit`, :meth:`score`, and the
    :attr:`detector_name` property.

    Attributes:
        threshold: Decision threshold; runs with ``anomaly_score >= threshold``
            are flagged as anomalies. Defaults to ``0.5``.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialise the detector.

        Args:
            threshold: Anomaly decision threshold in [0.0, 1.0].
        """
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def detector_name(self) -> str:
        """Human-readable identifier for this detector.

        Returns:
            A short lowercase string, e.g. ``"zscore"`` or
            ``"isolation_forest"``.
        """
        ...

    @abc.abstractmethod
    def fit(self, runs: list[PipelineRun]) -> "BaseDetector":
        """Train the detector on historical pipeline runs.

        Args:
            runs: Historical pipeline runs to train on.

        Returns:
            Self, for method chaining.
        """
        ...

    @abc.abstractmethod
    def score(self, run: PipelineRun) -> AnomalyScore:
        """Score a single pipeline run for anomalousness.

        Args:
            run: The pipeline run to score.

        Returns:
            An :class:`AnomalyScore` with ``anomaly_score`` in [0.0, 1.0].
        """
        ...

    # ------------------------------------------------------------------
    # Concrete methods
    # ------------------------------------------------------------------

    def batch_score(self, runs: list[PipelineRun]) -> list[AnomalyScore]:
        """Score multiple pipeline runs.

        The default implementation calls :meth:`score` for each run.
        Subclasses may override this for efficiency.

        Args:
            runs: Pipeline runs to score.

        Returns:
            List of :class:`AnomalyScore` objects, one per run.
        """
        scores: list[AnomalyScore] = []
        for run in runs:
            try:
                scores.append(self.score(run))
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "base_detector_score_error",
                    run_id=run.run_id,
                    detector=self.detector_name,
                    error=str(exc),
                )
        return scores
