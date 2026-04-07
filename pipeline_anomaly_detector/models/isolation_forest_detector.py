"""Isolation Forest based anomaly detector."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import structlog
from sklearn.ensemble import IsolationForest

from pipeline_anomaly_detector import PipelineRun
from pipeline_anomaly_detector.features.feature_extractor import FeatureExtractor
from pipeline_anomaly_detector.models.base_detector import AnomalyScore, BaseDetector

log = structlog.get_logger(__name__)


class IsolationForestDetector(BaseDetector):
    """Anomaly detector backed by scikit-learn's :class:`~sklearn.ensemble.IsolationForest`.

    Training procedure:

    1. Extracts features from all training runs via :class:`~pipeline_anomaly_detector.features.FeatureExtractor`.
    2. Splits data 90/10 into train and held-out sets.
    3. Fits :class:`~sklearn.ensemble.IsolationForest` on the training split.
    4. Computes permutation importance on the held-out set to rank features.

    Scoring:

    - Calls ``decision_function`` to get raw isolation scores.
    - Maps scores to [0, 1] via min-max normalisation using training-set bounds.
    - The inverted score (higher = more anomalous) is the ``anomaly_score``.

    Args:
        contamination: Expected fraction of anomalies in training data.
        threshold: Decision threshold for flagging anomalies.
        random_state: Seed for reproducibility.

    Example::

        detector = IsolationForestDetector(contamination=0.05, threshold=0.5)
        detector.fit(historical_runs)
        score = detector.score(new_run)
    """

    def __init__(
        self,
        contamination: float = 0.05,
        threshold: float = 0.5,
        random_state: int = 42,
    ) -> None:
        """Initialise the IsolationForestDetector.

        Args:
            contamination: Fraction of outliers expected in training data.
            threshold: Anomaly decision threshold.
            random_state: Random seed for :class:`~sklearn.ensemble.IsolationForest`.
        """
        super().__init__(threshold=threshold)
        self._contamination = contamination
        self._random_state = random_state
        self._extractor: FeatureExtractor | None = None
        self._model: IsolationForest | None = None
        self._feature_importances: dict[str, float] = {}
        self._feature_names: list[str] = []
        # Min/max of decision_function on training data for normalisation
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self._is_fitted: bool = False
        self._training_runs: list[PipelineRun] = []

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    @property
    def detector_name(self) -> str:
        """Return the detector's identifier.

        Returns:
            ``"isolation_forest"``
        """
        return "isolation_forest"

    def fit(self, runs: list[PipelineRun]) -> "IsolationForestDetector":
        """Fit the isolation forest on training runs.

        Splits the data 90/10 into train and held-out sets. If the held-out
        set has fewer than 5 samples the entire training set is used for
        permutation importance.

        Args:
            runs: Historical pipeline runs to train on.

        Returns:
            Self, for method chaining.
        """
        self._training_runs = list(runs)

        self._extractor = FeatureExtractor()
        feature_df = self._extractor.fit_transform(runs)

        self._feature_names = list(feature_df.columns)
        X = feature_df.fillna(0.0).values.astype(float)

        if len(X) == 0:
            log.warning("isolation_forest_no_data")
            return self

        # 90/10 train/held-out split
        n_total = len(X)
        n_train = max(int(n_total * 0.9), 1)
        X_train = X[:n_train]
        X_held = X[n_train:]

        self._model = IsolationForest(
            contamination=self._contamination,
            random_state=self._random_state,
            n_estimators=100,
        )
        self._model.fit(X_train)

        # Compute training score range for normalisation
        train_scores = self._model.decision_function(X_train)
        self._score_min = float(np.min(train_scores))
        self._score_max = float(np.max(train_scores))

        # Feature importances via permutation importance (manual, for unsupervised model).
        # We measure how much shuffling each feature column changes the mean
        # decision_function score on the evaluation set.  A larger drop indicates
        # greater importance.
        eval_X = X_held if len(X_held) >= 5 else X_train
        if len(eval_X) == 0:
            eval_X = X_train

        log.debug(
            "isolation_forest_computing_importance",
            n_eval=len(eval_X),
            n_held=len(X_held),
        )

        importances = self._permutation_importance_unsupervised(
            model=self._model,
            X=eval_X,
            n_repeats=5,
            random_state=self._random_state,
        )

        self._feature_importances = {
            name: float(imp)
            for name, imp in zip(self._feature_names, importances)
        }

        self._is_fitted = True
        log.info(
            "isolation_forest_fitted",
            n_runs=len(runs),
            n_train=n_train,
            n_held_out=len(X_held),
            feature_count=len(self._feature_names),
        )
        return self

    def score(self, run: PipelineRun) -> AnomalyScore:
        """Score a single pipeline run with the isolation forest.

        Args:
            run: The pipeline run to score.

        Returns:
            An :class:`~pipeline_anomaly_detector.models.base_detector.AnomalyScore`
            with normalised ``anomaly_score`` in [0.0, 1.0].

        Raises:
            RuntimeError: If the detector has not been fitted yet.
        """
        if not self._is_fitted or self._model is None or self._extractor is None:
            raise RuntimeError(
                "IsolationForestDetector must be fitted before scoring. "
                "Call fit() first."
            )

        # Add the current run to history for feature extraction context
        context_runs = self._training_runs + [run]
        temp_extractor = FeatureExtractor()
        temp_extractor.fit(self._training_runs)

        try:
            feature_df = temp_extractor.transform([run])
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "isolation_forest_feature_error",
                run_id=run.run_id,
                error=str(exc),
            )
            return self._zero_score(run)

        if feature_df.empty:
            return self._zero_score(run)

        # Align feature columns
        X = self._align_features(feature_df)
        raw_score = float(self._model.decision_function(X)[0])

        # Map to [0, 1]: higher raw score = more normal -> invert for anomaly
        score_range = self._score_max - self._score_min
        if score_range > 0:
            normalised = (raw_score - self._score_min) / score_range
        else:
            normalised = 0.5
        # Invert: isolation forest gives higher score to normals
        anomaly_score = float(np.clip(1.0 - normalised, 0.0, 1.0))

        # Contributing features: top 3 by permutation importance
        contributing = sorted(
            self._feature_importances.keys(),
            key=lambda k: self._feature_importances[k],
            reverse=True,
        )[:3]

        is_anomaly = anomaly_score >= self.threshold

        log.debug(
            "isolation_forest_scored",
            run_id=run.run_id,
            pipeline_name=run.pipeline_name,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
        )

        return AnomalyScore(
            run_id=run.run_id,
            pipeline_name=run.pipeline_name,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            contributing_features=contributing,
            detector_name=self.detector_name,
            timestamp=datetime.now(tz=timezone.utc),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _permutation_importance_unsupervised(
        model: IsolationForest,
        X: np.ndarray,
        n_repeats: int = 5,
        random_state: int = 42,
    ) -> np.ndarray:
        """Compute permutation feature importance for an unsupervised model.

        For each feature, shuffles its values ``n_repeats`` times and measures
        the mean decrease in the ``decision_function`` score.  A larger
        decrease indicates greater importance.

        Args:
            model: Fitted :class:`~sklearn.ensemble.IsolationForest`.
            X: Evaluation data array of shape ``(n_samples, n_features)``.
            n_repeats: Number of shuffle repeats per feature.
            random_state: Random seed for reproducibility.

        Returns:
            Array of shape ``(n_features,)`` with mean importance values.
        """
        rng = np.random.RandomState(random_state)
        baseline_scores = model.decision_function(X)
        baseline_mean = float(np.mean(baseline_scores))

        n_features = X.shape[1]
        importances = np.zeros(n_features, dtype=float)

        for feat_idx in range(n_features):
            drops = []
            for _ in range(n_repeats):
                X_shuffled = X.copy()
                rng.shuffle(X_shuffled[:, feat_idx])
                shuffled_mean = float(np.mean(model.decision_function(X_shuffled)))
                # A positive drop means the feature was important
                drops.append(baseline_mean - shuffled_mean)
            importances[feat_idx] = float(np.mean(drops))

        return importances

    def _align_features(self, feature_df) -> np.ndarray:
        """Align feature DataFrame columns to match training feature order.

        Args:
            feature_df: DataFrame of features for a single run.

        Returns:
            NumPy array with columns aligned to ``self._feature_names``.
        """
        aligned = []
        for name in self._feature_names:
            if name in feature_df.columns:
                val = feature_df[name].iloc[0]
                aligned.append(float(val) if not np.isnan(float(val)) else 0.0)
            else:
                aligned.append(0.0)
        return np.array(aligned, dtype=float).reshape(1, -1)

    def _zero_score(self, run: PipelineRun) -> AnomalyScore:
        """Return a zero-score :class:`AnomalyScore` for error cases.

        Args:
            run: The pipeline run that failed to score.

        Returns:
            An :class:`AnomalyScore` with ``anomaly_score=0.0``.
        """
        return AnomalyScore(
            run_id=run.run_id,
            pipeline_name=run.pipeline_name,
            anomaly_score=0.0,
            is_anomaly=False,
            contributing_features=[],
            detector_name=self.detector_name,
            timestamp=datetime.now(tz=timezone.utc),
        )
