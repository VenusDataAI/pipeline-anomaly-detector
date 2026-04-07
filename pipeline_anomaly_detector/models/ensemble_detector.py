"""Ensemble anomaly detector that combines multiple sub-detectors."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from pipeline_anomaly_detector import PipelineRun
from pipeline_anomaly_detector.models.base_detector import AnomalyScore, BaseDetector

log = structlog.get_logger(__name__)


class EnsembleDetector(BaseDetector):
    """Weighted ensemble of multiple anomaly detectors.

    Combines the anomaly scores from each sub-detector via a weighted average.
    Contributing features are the deduplicated union of all sub-detector
    contributing features, preserving insertion order.

    Args:
        detectors: List of fitted or unfitted :class:`~pipeline_anomaly_detector.models.base_detector.BaseDetector`
            instances to combine.
        weights: Optional list of floats (same length as *detectors*).
            Will be normalised to sum to 1.0 if necessary.
            If ``None``, equal weights are used.
        threshold: Decision threshold; runs with aggregated
            ``anomaly_score >= threshold`` are flagged. Defaults to ``0.6``.

    Example::

        detectors = [ZScoreDetector(), IsolationForestDetector()]
        ensemble = EnsembleDetector(detectors=detectors, threshold=0.6)
        ensemble.fit(historical_runs)
        score = ensemble.score(new_run)
    """

    def __init__(
        self,
        detectors: list[BaseDetector],
        weights: list[float] | None = None,
        threshold: float = 0.6,
    ) -> None:
        """Initialise the ensemble detector.

        Args:
            detectors: Sub-detectors to combine.
            weights: Optional per-detector weights. Normalised to sum to 1.0.
            threshold: Decision threshold for flagging anomalies.

        Raises:
            ValueError: If *detectors* is empty.
        """
        super().__init__(threshold=threshold)

        if not detectors:
            raise ValueError("EnsembleDetector requires at least one sub-detector.")

        self._detectors = detectors
        self._weights = self._normalise_weights(weights, len(detectors))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_weights(
        weights: list[float] | None, n: int
    ) -> list[float]:
        """Normalise or default weights to sum to 1.0.

        Args:
            weights: Raw weights or ``None`` for equal weights.
            n: Number of detectors.

        Returns:
            Normalised list of floats summing to 1.0.

        Raises:
            ValueError: If *weights* has a different length than *n* or all
                values are zero.
        """
        if weights is None:
            w = 1.0 / n
            return [w] * n

        if len(weights) != n:
            raise ValueError(
                f"Expected {n} weights, got {len(weights)}."
            )

        total = sum(weights)
        if total <= 0.0:
            raise ValueError("Weights must sum to a positive value.")

        return [w / total for w in weights]

    @staticmethod
    def _deduplicate_features(feature_lists: list[list[str]]) -> list[str]:
        """Merge feature lists, preserving first-seen order without duplicates.

        Args:
            feature_lists: Lists of feature name strings from each detector.

        Returns:
            Deduplicated list of feature names.
        """
        seen: set[str] = set()
        result: list[str] = []
        for features in feature_lists:
            for feat in features:
                if feat not in seen:
                    seen.add(feat)
                    result.append(feat)
        return result

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    @property
    def detector_name(self) -> str:
        """Return the detector's identifier.

        Returns:
            ``"ensemble"``
        """
        return "ensemble"

    def fit(self, runs: list[PipelineRun]) -> "EnsembleDetector":
        """Fit all sub-detectors on the same training runs.

        Args:
            runs: Historical pipeline runs.

        Returns:
            Self, for method chaining.
        """
        for i, detector in enumerate(self._detectors):
            log.info(
                "ensemble_fitting_sub_detector",
                index=i,
                name=detector.detector_name,
                n_runs=len(runs),
            )
            detector.fit(runs)

        log.info(
            "ensemble_fitted",
            n_detectors=len(self._detectors),
            weights=self._weights,
        )
        return self

    def score(self, run: PipelineRun) -> AnomalyScore:
        """Score a run by computing the weighted average of sub-detector scores.

        Args:
            run: The pipeline run to score.

        Returns:
            An :class:`~pipeline_anomaly_detector.models.base_detector.AnomalyScore`
            with the weighted-average ``anomaly_score`` and the union of
            sub-detector contributing features.
        """
        sub_scores = [detector.score(run) for detector in self._detectors]

        weighted_score = sum(
            w * s.anomaly_score
            for w, s in zip(self._weights, sub_scores)
        )
        weighted_score = max(0.0, min(1.0, weighted_score))

        contributing = self._deduplicate_features(
            [s.contributing_features for s in sub_scores]
        )

        is_anomaly = weighted_score >= self.threshold

        log.debug(
            "ensemble_scored",
            run_id=run.run_id,
            pipeline_name=run.pipeline_name,
            anomaly_score=weighted_score,
            is_anomaly=is_anomaly,
            sub_scores=[s.anomaly_score for s in sub_scores],
        )

        return AnomalyScore(
            run_id=run.run_id,
            pipeline_name=run.pipeline_name,
            anomaly_score=weighted_score,
            is_anomaly=is_anomaly,
            contributing_features=contributing,
            detector_name=self.detector_name,
            timestamp=datetime.now(tz=timezone.utc),
        )

    def batch_score(self, runs: list[PipelineRun]) -> list[AnomalyScore]:
        """Score multiple runs by aggregating sub-detector batch scores.

        Args:
            runs: Pipeline runs to score.

        Returns:
            List of :class:`~pipeline_anomaly_detector.models.base_detector.AnomalyScore`
            objects.
        """
        if not runs:
            return []

        # Collect batch scores from each sub-detector
        all_sub_scores: list[list[AnomalyScore]] = [
            detector.batch_score(runs) for detector in self._detectors
        ]

        results: list[AnomalyScore] = []
        for i, run in enumerate(runs):
            sub_scores_for_run = [
                all_sub_scores[j][i]
                for j in range(len(self._detectors))
                if i < len(all_sub_scores[j])
            ]

            if not sub_scores_for_run:
                continue

            weighted_score = sum(
                w * s.anomaly_score
                for w, s in zip(self._weights, sub_scores_for_run)
            )
            weighted_score = max(0.0, min(1.0, weighted_score))

            contributing = self._deduplicate_features(
                [s.contributing_features for s in sub_scores_for_run]
            )

            is_anomaly = weighted_score >= self.threshold

            results.append(
                AnomalyScore(
                    run_id=run.run_id,
                    pipeline_name=run.pipeline_name,
                    anomaly_score=weighted_score,
                    is_anomaly=is_anomaly,
                    contributing_features=contributing,
                    detector_name=self.detector_name,
                    timestamp=datetime.now(tz=timezone.utc),
                )
            )

        return results
