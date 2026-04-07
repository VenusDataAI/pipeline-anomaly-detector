"""Airflow metadata database collector."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from pipeline_anomaly_detector import PipelineRun
from pipeline_anomaly_detector.collectors.base_collector import BaseCollector

log = structlog.get_logger(__name__)

try:
    from airflow.models import DagRun, TaskInstance  # type: ignore[import]

    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False
    log.debug("airflow_not_installed", message="AirflowCollector will raise if used")


_AIRFLOW_STATUS_MAP: dict[str, str] = {
    "success": "success",
    "failed": "failed",
    "upstream_failed": "failed",
    "skipped": "skipped",
    "removed": "skipped",
}


class AirflowCollector(BaseCollector):
    """Collect pipeline runs from an Airflow metadata database.

    Queries ``dag_run`` and ``task_instance`` tables via SQLAlchemy and
    maps each task execution to a
    :class:`~pipeline_anomaly_detector.PipelineRun`.

    Supports both **SQLite** (local development) and **PostgreSQL**
    (production) connection strings.

    Args:
        db_url: SQLAlchemy connection URL, e.g.
            ``"sqlite:///airflow.db"`` or
            ``"postgresql+psycopg2://user:pw@host/airflow"``.
        dag_ids: Optional list of DAG IDs to filter. If ``None``, all DAGs
            are included.

    Raises:
        ImportError: If ``apache-airflow`` is not installed **and** this
            collector is actually used (i.e. :meth:`collect` is called).

    Example::

        collector = AirflowCollector("sqlite:///airflow.db")
        runs = collector.collect(since=datetime(2024, 1, 1, tzinfo=timezone.utc))
    """

    def __init__(
        self,
        db_url: str,
        dag_ids: list[str] | None = None,
    ) -> None:
        """Initialise the Airflow collector.

        Args:
            db_url: SQLAlchemy database connection URL.
            dag_ids: Optional allowlist of DAG IDs. Queries all DAGs when
                ``None``.
        """
        self._db_url = db_url
        self._dag_ids = dag_ids
        log.debug(
            "airflow_collector_initialised",
            db_url=db_url,
            dag_ids=dag_ids,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_engine(self) -> Any:
        """Create and return a SQLAlchemy engine.

        Returns:
            A :class:`sqlalchemy.engine.Engine` instance.
        """
        from sqlalchemy import create_engine

        return create_engine(self._db_url)

    @staticmethod
    def _ensure_tz(dt: datetime | None) -> datetime | None:
        """Attach UTC timezone to a naive datetime if necessary.

        Args:
            dt: A potentially naive datetime.

        Returns:
            A timezone-aware datetime, or ``None`` if *dt* is ``None``.
        """
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    @staticmethod
    def _map_status(state: str | None) -> str:
        """Map an Airflow task state to a PipelineRun status.

        Args:
            state: Airflow task state string.

        Returns:
            One of ``"success"``, ``"failed"``, or ``"skipped"``.
        """
        return _AIRFLOW_STATUS_MAP.get(str(state or "").lower(), "failed")

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    def collect(self, since: datetime) -> list[PipelineRun]:
        """Query the Airflow metadata DB and return task-level pipeline runs.

        Each ``task_instance`` row is mapped to a single
        :class:`~pipeline_anomaly_detector.PipelineRun` where
        ``pipeline_name`` is ``"<dag_id>.<task_id>"``.

        Args:
            since: Only task instances that started at or after this
                timestamp are returned.

        Returns:
            List of :class:`~pipeline_anomaly_detector.PipelineRun` objects
            sorted by ``start_time`` ascending.

        Raises:
            ImportError: If ``apache-airflow`` is not installed.
            sqlalchemy.exc.SQLAlchemyError: If the database query fails.
        """
        if not _AIRFLOW_AVAILABLE:
            raise ImportError(
                "apache-airflow is not installed. "
                "Install it with: pip install pipeline-anomaly-detector[airflow]"
            )

        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)

        from sqlalchemy import text

        engine = self._get_engine()
        runs: list[PipelineRun] = []

        dag_filter = ""
        params: dict[str, Any] = {"since": since}

        if self._dag_ids:
            placeholders = ", ".join(f":dag_{i}" for i in range(len(self._dag_ids)))
            dag_filter = f"AND ti.dag_id IN ({placeholders})"
            for i, dag_id in enumerate(self._dag_ids):
                params[f"dag_{i}"] = dag_id

        query = text(
            f"""
            SELECT
                ti.dag_id,
                ti.task_id,
                ti.run_id,
                ti.start_date,
                ti.end_date,
                ti.duration,
                ti.state,
                dr.execution_date
            FROM task_instance ti
            JOIN dag_run dr
                ON ti.dag_id = dr.dag_id
                AND ti.run_id = dr.run_id
            WHERE ti.start_date >= :since
              {dag_filter}
            ORDER BY ti.start_date ASC
            """
        )

        with engine.connect() as conn:
            rows = conn.execute(query, params).fetchall()

        log.debug("airflow_collector_rows_fetched", count=len(rows))

        for row in rows:
            dag_id = row[0]
            task_id = row[1]
            airflow_run_id = row[2]
            start_date = self._ensure_tz(row[3])
            end_date = self._ensure_tz(row[4])
            duration_raw = row[5]
            state = row[6]

            if start_date is None:
                continue

            # Compute duration
            if isinstance(duration_raw, (int, float)):
                duration = float(duration_raw)
            elif end_date is not None and start_date is not None:
                duration = max(0.0, (end_date - start_date).total_seconds())
            else:
                duration = 0.0

            end_date_final = end_date or datetime.fromtimestamp(
                start_date.timestamp() + duration, tz=timezone.utc
            )

            pipeline_name = f"{dag_id}.{task_id}"
            run_id = f"airflow_{dag_id}_{task_id}_{airflow_run_id}"
            status = self._map_status(state)

            try:
                run = PipelineRun(
                    run_id=run_id,
                    pipeline_name=pipeline_name,
                    start_time=start_date,
                    end_time=end_date_final,
                    duration_seconds=max(0.0, duration),
                    rows_processed=0,
                    rows_rejected=0,
                    null_rate={},
                    status=status,  # type: ignore[arg-type]
                    metadata={
                        "dag_id": dag_id,
                        "task_id": task_id,
                        "airflow_run_id": airflow_run_id,
                        "state": str(state),
                    },
                )
                runs.append(run)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "airflow_collector_run_creation_error",
                    dag_id=dag_id,
                    task_id=task_id,
                    error=str(exc),
                )

        log.info(
            "airflow_collector_collected",
            total_rows=len(rows),
            valid_runs=len(runs),
        )
        return runs
