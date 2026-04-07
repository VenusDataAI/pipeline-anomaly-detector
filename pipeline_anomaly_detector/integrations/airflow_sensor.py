"""Airflow sensor operator for anomaly detection."""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

try:
    from airflow.sensors.base import BaseSensorOperator  # type: ignore[import]
    from airflow.exceptions import AirflowException  # type: ignore[import]

    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False

    class BaseSensorOperator:  # type: ignore[no-redef]
        """Stub base class used when airflow is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            pass

    class AirflowException(Exception):  # type: ignore[no-redef]
        """Stub exception used when airflow is not installed."""
        pass


class AnomalyDetectionSensor(BaseSensorOperator):
    """Airflow sensor that pokes for pipeline anomalies.

    Loads a serialised :class:`~pipeline_anomaly_detector.models.BaseDetector`
    from disk, scores the most recent pipeline run, and signals Airflow
    accordingly:

    - Returns ``False`` (keep poking) when the anomaly score is below the
      threshold.
    - Raises :class:`~airflow.exceptions.AirflowException` when an anomaly is
      detected, causing the sensor task to fail and trigger downstream alerts.

    Args:
        pipeline_name: Name of the pipeline to monitor.
        detector_path: Filesystem path to the ``.joblib`` detector file.
        db_path: Path to the SQLite anomaly scores database.
        anomaly_threshold: Score threshold above which an anomaly is flagged.
        **kwargs: Additional keyword arguments forwarded to
            :class:`~airflow.sensors.base.BaseSensorOperator`.

    Example (in an Airflow DAG)::

        sensor = AnomalyDetectionSensor(
            task_id="check_orders_anomaly",
            pipeline_name="orders_pipeline",
            detector_path="/models/global_isolation_forest_latest.joblib",
            db_path="/data/anomaly_scores.db",
            poke_interval=300,
        )
    """

    def __init__(
        self,
        pipeline_name: str,
        detector_path: str,
        db_path: str = "anomaly_scores.db",
        anomaly_threshold: float = 0.5,
        **kwargs,
    ) -> None:
        """Initialise the sensor.

        Args:
            pipeline_name: Pipeline name to monitor.
            detector_path: Path to the serialised detector ``.joblib`` file.
            db_path: SQLite database path for anomaly score history.
            anomaly_threshold: Anomaly decision threshold.
            **kwargs: Forwarded to the Airflow base sensor.
        """
        super().__init__(**kwargs)
        self.pipeline_name = pipeline_name
        self.detector_path = detector_path
        self.db_path = db_path
        self.anomaly_threshold = anomaly_threshold

    def poke(self, context) -> bool:
        """Poke: load the detector, score the latest run, return status.

        Args:
            context: Airflow task context dict.

        Returns:
            ``False`` if the anomaly score is below the threshold (sensor
            keeps poking). Raises :class:`AirflowException` if an anomaly is
            detected.

        Raises:
            AirflowException: When ``anomaly_score >= anomaly_threshold``.
        """
        from pipeline_anomaly_detector.training.model_store import ModelStore
        from pipeline_anomaly_detector.scoring.scorer import Scorer

        log.info(
            "anomaly_sensor_poke",
            pipeline_name=self.pipeline_name,
            detector_path=self.detector_path,
        )

        try:
            # Load the detector
            store = ModelStore(store_dir=str(self.detector_path).rsplit("/", 1)[0])
            detector = store.load(self.detector_path)

            # Get recent scores from the database
            scorer = Scorer(detector=detector, db_path=self.db_path)
            history = scorer.get_history(self.pipeline_name, limit=1)

            if not history:
                log.info(
                    "anomaly_sensor_no_history",
                    pipeline_name=self.pipeline_name,
                )
                return False

            latest = history[0]
            score_value = latest.get("anomaly_score", 0.0)
            is_anomaly = latest.get("is_anomaly", False)

            if is_anomaly or score_value >= self.anomaly_threshold:
                msg = (
                    f"Anomaly detected in pipeline '{self.pipeline_name}': "
                    f"score={score_value:.4f}, run_id={latest.get('run_id')}"
                )
                log.warning(
                    "anomaly_sensor_anomaly_detected",
                    pipeline_name=self.pipeline_name,
                    score=score_value,
                )
                raise AirflowException(msg)

            log.debug(
                "anomaly_sensor_normal",
                pipeline_name=self.pipeline_name,
                score=score_value,
                threshold=self.anomaly_threshold,
            )
            return False

        except AirflowException:
            raise
        except Exception as exc:  # noqa: BLE001
            log.error(
                "anomaly_sensor_error",
                error=str(exc),
                pipeline_name=self.pipeline_name,
            )
            return False
