"""Feature extraction from PipelineRun objects."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import structlog

from pipeline_anomaly_detector import PipelineRun
from pipeline_anomaly_detector.features.feature_registry import FEATURE_REGISTRY

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)


class FeatureExtractor:
    """Extract numerical features from :class:`~pipeline_anomaly_detector.PipelineRun` objects.

    The extractor maintains per-pipeline historical run data so that
    history-dependent features (z-scores, delta percentages) can be computed.

    Usage::

        extractor = FeatureExtractor()
        # Optionally fit on historical data first:
        extractor.fit(historical_runs)
        # Then transform new runs:
        df = extractor.transform(new_runs)

    The returned DataFrame has ``run_id`` as the index and one column per
    feature (11 total, matching :data:`~pipeline_anomaly_detector.features.FEATURE_REGISTRY`).
    """

    def __init__(self, window: int = 30) -> None:
        """Initialise the extractor.

        Args:
            window: Number of historical runs per pipeline to use for rolling
                statistics. Defaults to 30.
        """
        self._window = window
        # pipeline_name -> list of PipelineRun (ordered by start_time)
        self._history: dict[str, list[PipelineRun]] = defaultdict(list)
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, runs: list[PipelineRun]) -> "FeatureExtractor":
        """Store historical runs for use in windowed features.

        Replaces any previously stored history.

        Args:
            runs: Historical pipeline runs. They are grouped by
                ``pipeline_name`` and sorted by ``start_time``.

        Returns:
            Self, for method chaining.
        """
        self._history = defaultdict(list)
        for run in sorted(runs, key=lambda r: r.start_time):
            self._history[run.pipeline_name].append(run)

        self._is_fitted = True
        log.info(
            "feature_extractor_fitted",
            n_runs=len(runs),
            n_pipelines=len(self._history),
        )
        return self

    def transform(self, runs: list[PipelineRun]) -> pd.DataFrame:
        """Extract all 11 features for the given runs.

        When history-dependent features cannot be computed (no prior runs or
        insufficient history), they degrade gracefully to ``0.0`` rather than
        raising exceptions.

        Args:
            runs: Pipeline runs to featurise.

        Returns:
            A :class:`pandas.DataFrame` with ``run_id`` as the index and
            feature columns matching :data:`~pipeline_anomaly_detector.features.FEATURE_REGISTRY`.
        """
        records: list[dict] = []
        for run in runs:
            record = self._extract_single(run)
            records.append(record)

        if not records:
            return pd.DataFrame(
                columns=["run_id"] + FEATURE_REGISTRY.feature_names
            ).set_index("run_id")

        df = pd.DataFrame(records).set_index("run_id")
        # Enforce dtypes from registry
        for spec in FEATURE_REGISTRY:
            if spec.name in df.columns:
                try:
                    df[spec.name] = df[spec.name].astype(spec.dtype)
                except (ValueError, TypeError):
                    pass
        return df

    def fit_transform(self, runs: list[PipelineRun]) -> pd.DataFrame:
        """Fit on *runs* and then transform them.

        Equivalent to calling :meth:`fit` followed by :meth:`transform` with
        the same runs.

        Args:
            runs: Pipeline runs to fit on and featurise.

        Returns:
            Feature :class:`pandas.DataFrame` with ``run_id`` as index.
        """
        self.fit(runs)
        return self.transform(runs)

    # ------------------------------------------------------------------
    # Internal: rolling statistics
    # ------------------------------------------------------------------

    def _compute_rolling_stats(
        self, pipeline_name: str, window: int | None = None
    ) -> tuple[float, float]:
        """Compute mean and std of ``duration_seconds`` over recent history.

        Uses up to *window* most recent historical runs for the given pipeline.
        When fewer runs are available the method degrades gracefully by using
        whatever history exists.

        Args:
            pipeline_name: Pipeline to compute statistics for.
            window: Number of recent runs to consider. Defaults to
                :attr:`_window`.

        Returns:
            A ``(mean, std)`` tuple. Returns ``(0.0, 0.0)`` if no history is
            available. Returns ``(mean, 0.0)`` if only one run is available
            (std is undefined).
        """
        w = window or self._window
        history = self._history.get(pipeline_name, [])
        if len(history) == 0:
            return 0.0, 0.0

        recent = history[-w:]
        durations = np.array([r.duration_seconds for r in recent], dtype=float)
        mean = float(np.mean(durations))
        std = float(np.std(durations, ddof=1)) if len(durations) > 1 else 0.0
        return mean, std

    # ------------------------------------------------------------------
    # Internal: single-run feature computation
    # ------------------------------------------------------------------

    def _extract_single(self, run: PipelineRun) -> dict:
        """Compute all 11 features for a single pipeline run.

        Args:
            run: The pipeline run to featurise.

        Returns:
            A dict mapping feature names (plus ``"run_id"``) to scalar values.
        """
        pipeline_name = run.pipeline_name
        history = self._history.get(pipeline_name, [])

        # ---- No-history features ----------------------------------------

        duration_seconds = float(run.duration_seconds)

        rows_processed_log1p = float(np.log1p(run.rows_processed))

        null_rate_max = (
            float(max(run.null_rate.values())) if run.null_rate else 0.0
        )

        hour_of_day = int(run.start_time.hour)
        day_of_week = int(run.start_time.weekday())
        is_weekend = int(run.start_time.weekday() >= 5)
        status_is_success = int(run.status == "success")

        rejection_rate = (
            run.rows_rejected / run.rows_processed
            if run.rows_processed > 0
            else 0.0
        )

        # ---- History-dependent features ---------------------------------

        # duration_z
        mean_dur, std_dur = self._compute_rolling_stats(pipeline_name)
        if std_dur > 0.0:
            duration_z = (duration_seconds - mean_dur) / std_dur
        else:
            duration_z = 0.0

        # row_count_delta_pct and null_rate_delta: use most recent prior run
        prior_runs = [r for r in history if r.run_id != run.run_id]
        if prior_runs:
            # Sort and take the most recent
            prior_runs_sorted = sorted(prior_runs, key=lambda r: r.start_time)
            prev_run = prior_runs_sorted[-1]

            prev_rows = prev_run.rows_processed
            if prev_rows > 0:
                row_count_delta_pct = (
                    (run.rows_processed - prev_rows) / prev_rows * 100.0
                )
            else:
                row_count_delta_pct = 0.0

            prev_null_max = (
                float(max(prev_run.null_rate.values()))
                if prev_run.null_rate
                else 0.0
            )
            null_rate_delta = null_rate_max - prev_null_max
        else:
            row_count_delta_pct = 0.0
            null_rate_delta = 0.0

        return {
            "run_id": run.run_id,
            "duration_seconds": duration_seconds,
            "duration_z": duration_z,
            "rows_processed_log1p": rows_processed_log1p,
            "row_count_delta_pct": row_count_delta_pct,
            "null_rate_max": null_rate_max,
            "null_rate_delta": null_rate_delta,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "status_is_success": status_is_success,
            "rejection_rate": rejection_rate,
        }
