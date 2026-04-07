"""Scorer: scores pipeline runs, persists results, and routes alerts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import structlog
from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session

from pipeline_anomaly_detector import PipelineRun
from pipeline_anomaly_detector.models.base_detector import AnomalyScore, BaseDetector

log = structlog.get_logger(__name__)


class _Base(DeclarativeBase):
    pass


class _AnomalyScoreRow(_Base):
    """SQLAlchemy ORM model for the anomaly_scores table."""

    __tablename__ = "anomaly_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False, index=True)
    pipeline_name = Column(String, nullable=False, index=True)
    anomaly_score = Column(Float, nullable=False)
    is_anomaly = Column(Boolean, nullable=False)
    contributing_features = Column(Text, nullable=True)  # JSON-encoded list
    detector_name = Column(String, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)


class Scorer:
    """Scores pipeline runs using a detector, persists results, and routes alerts.

    Args:
        detector: A fitted :class:`~pipeline_anomaly_detector.models.BaseDetector`.
        db_path: Path to the SQLite database file. Defaults to
            ``"anomaly_scores.db"`` in the current directory.
        alert_router: Optional :class:`~pipeline_anomaly_detector.scoring.AlertRouter`
            to call when an anomaly is detected.

    Example::

        scorer = Scorer(detector=detector, db_path="scores.db")
        score = scorer.score(run)
        history = scorer.get_history("orders_pipeline", limit=10)
    """

    def __init__(
        self,
        detector: BaseDetector,
        db_path: Union[str, Path] = "anomaly_scores.db",
        alert_router=None,
    ) -> None:
        """Initialise the scorer.

        Args:
            detector: Fitted anomaly detector.
            db_path: SQLite database path.
            alert_router: Optional alert router for anomaly notifications.
        """
        self._detector = detector
        self._db_path = Path(db_path)
        self._alert_router = alert_router
        self._engine = create_engine(
            f"sqlite:///{self._db_path}",
            connect_args={"check_same_thread": False},
        )
        self._init_db()
        log.info(
            "scorer_initialised",
            detector=detector.detector_name,
            db_path=str(self._db_path),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, run: PipelineRun) -> AnomalyScore:
        """Score a single pipeline run.

        The result is persisted to the database. If the score indicates an
        anomaly and an alert router is configured, the alert is routed.

        Args:
            run: The pipeline run to score.

        Returns:
            The computed :class:`~pipeline_anomaly_detector.models.AnomalyScore`.
        """
        anomaly_score = self._detector.score(run)
        self._persist(anomaly_score)

        if anomaly_score.is_anomaly and self._alert_router is not None:
            self._alert_router.route(anomaly_score)

        log.debug(
            "scorer_scored",
            run_id=run.run_id,
            pipeline_name=run.pipeline_name,
            anomaly_score=anomaly_score.anomaly_score,
            is_anomaly=anomaly_score.is_anomaly,
        )
        return anomaly_score

    def score_batch(self, runs: list[PipelineRun]) -> list[AnomalyScore]:
        """Score a batch of pipeline runs.

        Each result is persisted and alerts are routed for anomalies.

        Args:
            runs: Pipeline runs to score.

        Returns:
            List of :class:`~pipeline_anomaly_detector.models.AnomalyScore`
            objects.
        """
        scores = [self.score(run) for run in runs]
        log.info(
            "scorer_batch_scored",
            n_runs=len(runs),
            n_anomalies=sum(1 for s in scores if s.is_anomaly),
        )
        return scores

    def get_history(self, pipeline_name: str, limit: int = 30) -> list[dict]:
        """Retrieve recent anomaly scores for a pipeline.

        Args:
            pipeline_name: The pipeline to query.
            limit: Maximum number of rows to return (most recent first).

        Returns:
            List of score dicts ordered by ``timestamp`` descending.
        """
        with Session(self._engine) as session:
            rows = (
                session.query(_AnomalyScoreRow)
                .filter(_AnomalyScoreRow.pipeline_name == pipeline_name)
                .order_by(_AnomalyScoreRow.timestamp.desc())
                .limit(limit)
                .all()
            )

        results = []
        for row in rows:
            try:
                contributing = json.loads(row.contributing_features or "[]")
            except (json.JSONDecodeError, TypeError):
                contributing = []
            results.append(
                {
                    "id": row.id,
                    "run_id": row.run_id,
                    "pipeline_name": row.pipeline_name,
                    "anomaly_score": row.anomaly_score,
                    "is_anomaly": row.is_anomaly,
                    "contributing_features": contributing,
                    "detector_name": row.detector_name,
                    "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the anomaly_scores table if it does not already exist."""
        _Base.metadata.create_all(self._engine)
        log.debug("scorer_db_initialised", db_path=str(self._db_path))

    def _persist(self, score: AnomalyScore) -> None:
        """Insert a scored result into the database.

        Args:
            score: The anomaly score to persist.
        """
        row = _AnomalyScoreRow(
            run_id=score.run_id,
            pipeline_name=score.pipeline_name,
            anomaly_score=score.anomaly_score,
            is_anomaly=score.is_anomaly,
            contributing_features=json.dumps(score.contributing_features),
            detector_name=score.detector_name,
            timestamp=score.timestamp,
        )
        with Session(self._engine) as session:
            session.add(row)
            session.commit()

        log.debug(
            "scorer_persisted",
            run_id=score.run_id,
            anomaly_score=score.anomaly_score,
        )
