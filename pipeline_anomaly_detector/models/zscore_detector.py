"""Z-score based anomaly detector."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import structlog

from pipeline_anomaly_detector import PipelineRun
from pipeline_anomaly_detector.features.feature_extractor import FeatureExtractor
from pipeline_anomaly_detector.models.base_detector import AnomalyScore, BaseDetector

log = structlog.get_logger(__name__)


class ZScoreDetector(BaseDetector):
    """Anomaly detector based on per-feature z-scores against a rolling window.

    For each pipeline run the detector:

    1. Extracts all 11 features via :class:`~pipeline_anomaly_detector.features.FeatureExtractor`.
    2. Computes per-feature z-scores against the rolling window of historical
       runs for the same pipeline.
    3. Maps the maximum absolute z-score to a normalised ``anomaly_score``
       via ``min(max_abs_z / 10.0, 1.0)``.
    4. Identifies ``contributing_features`` as those with ``|z| > 2.0``,
       sorted by descending ``|z|``, top 3.

    Args:
        window: Number of recent runs per pipeline to use for rolling stats.
        threshold: Decision threshold for flagging anomalies.

    Example::

        detector = ZScoreDetector(window=30, threshold=0.5)
        detector.fit(historical_runs)
        score = detector.score(new_run)
    """

    def __init__(self, window: int = 30, threshold: float = 0.5) -> None:
        """Initialise the ZScoreDetector.

        Args:
            window: Rolling window size for per-pipeline statistics.
            threshold: Anomaly decision threshold.
        """
        super().__init__(threshold=threshold)
        self._window = window
        # pipeline_name -> list[PipelineRun] sorted by start_time
        self._history: dict[str, list[PipelineRun]] = defaultdict(list)
        self._extractor = FeatureExtractor(window=window)
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    @property
    def detector_name(self) -> str:
        """Return the detector's identifier.

        Returns:
            ``"zscore"``
        """
        return "zscore"

    def fit(self, runs: list[PipelineRun]) -> "ZScoreDetector":
        """Fit the detector by storing historical runs per pipeline.

        Args:
            runs: Historical pipeline runs.

        Returns:
            Self, for method chaining.
        """
        self._history = defaultdict(list)
        for run in sorted(runs, key=lambda r: r.start_time):
            self._history[run.pipeline_name].append(run)

        self._extractor = FeatureExtractor(window=self._window)
        self._extractor.fit(runs)
        self._is_fitted = True

        log.info(
            "zscore_detector_fitted",
            n_runs=len(runs),
            n_pipelines=len(self._history),
            window=self._window,
        )
        return self

    def score(self, run: PipelineRun) -> AnomalyScore:
        """Score a single pipeline run using z-scores.

        The run is temporarily added to the per-pipeline history for context
        so that history-dependent features can be computed. It is then
        removed again.

        Args:
            run: The pipeline run to score.

        Returns:
            An :class:`~pipeline_anomaly_detector.models.base_detector.AnomalyScore`
            with normalised ``anomaly_score`` in [0.0, 1.0].
        """
        pipeline_name = run.pipeline_name
        history = self._history.get(pipeline_name, [])

        # Build a combined list: history + current run for feature extraction
        combined = list(history) + [run]

        # Build a temporary extractor with the combined history
        temp_extractor = FeatureExtractor(window=self._window)
        # Fit on historical only (excluding the run being scored)
        temp_extractor.fit(history)

        # Extract features for the current run only
        try:
            features_df = temp_extractor.transform([run])
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "zscore_detector_feature_error",
                run_id=run.run_id,
                error=str(exc),
            )
            return AnomalyScore(
                run_id=run.run_id,
                pipeline_name=pipeline_name,
                anomaly_score=0.0,
                is_anomaly=False,
                contributing_features=[],
                detector_name=self.detector_name,
                timestamp=datetime.now(tz=timezone.utc),
            )

        if features_df.empty:
            return AnomalyScore(
                run_id=run.run_id,
                pipeline_name=pipeline_name,
                anomaly_score=0.0,
                is_anomaly=False,
                contributing_features=[],
                detector_name=self.detector_name,
                timestamp=datetime.now(tz=timezone.utc),
            )

        # Compute per-feature z-scores against the rolling window of history
        feature_zscores: dict[str, float] = {}

        if len(history) >= 2:
            # Extract features for the historical window
            hist_window = history[-self._window:]
            hist_df = temp_extractor.transform(hist_window)

            for col in features_df.columns:
                if col not in hist_df.columns:
                    continue
                hist_vals = hist_df[col].dropna().astype(float).values
                if len(hist_vals) < 2:
                    feature_zscores[col] = 0.0
                    continue
                mean_h = float(np.mean(hist_vals))
                std_h = float(np.std(hist_vals, ddof=1))
                cur_val = float(features_df[col].iloc[0])
                if std_h > 0.0:
                    feature_zscores[col] = (cur_val - mean_h) / std_h
                else:
                    feature_zscores[col] = 0.0
        else:
            # Not enough history - use duration_z from feature extractor if available
            for col in features_df.columns:
                if col == "duration_z":
                    feature_zscores[col] = float(features_df[col].iloc[0])
                else:
                    feature_zscores[col] = 0.0

        # Compute anomaly score
        abs_zscores = {k: abs(v) for k, v in feature_zscores.items()}
        max_abs_z = max(abs_zscores.values()) if abs_zscores else 0.0
        anomaly_score = min(max_abs_z / 10.0, 1.0)

        # Contributing features: |z| > 2.0, sorted desc, top 3
        contributing = sorted(
            [k for k, v in abs_zscores.items() if v > 2.0],
            key=lambda k: abs_zscores[k],
            reverse=True,
        )[:3]

        is_anomaly = anomaly_score >= self.threshold

        log.debug(
            "zscore_detector_scored",
            run_id=run.run_id,
            pipeline_name=pipeline_name,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            max_abs_z=max_abs_z,
        )

        return AnomalyScore(
            run_id=run.run_id,
            pipeline_name=pipeline_name,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            contributing_features=contributing,
            detector_name=self.detector_name,
            timestamp=datetime.now(tz=timezone.utc),
        )
