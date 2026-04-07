"""Persistent storage for trained detector models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import joblib
import structlog

from pipeline_anomaly_detector.models.base_detector import BaseDetector

log = structlog.get_logger(__name__)

_METADATA_SUFFIX = ".meta.json"


class ModelStore:
    """Filesystem-backed store for serialised detector models.

    Models are saved as ``{pipeline_name}_{detector_name}_{timestamp}.joblib``
    alongside a JSON metadata sidecar file.

    Args:
        store_dir: Directory in which to save and load model files.

    Example::

        store = ModelStore("./models")
        path = store.save(detector, pipeline_name="orders_pipeline")
        loaded = store.load(path)
        latest = store.load_latest(pipeline_name="orders_pipeline")
    """

    def __init__(self, store_dir: Union[str, Path]) -> None:
        """Initialise the model store.

        Args:
            store_dir: Path to the directory for model artefacts. Created if
                it does not already exist.
        """
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        log.debug("model_store_initialised", store_dir=str(self._store_dir))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        detector: BaseDetector,
        pipeline_name: str = "global",
    ) -> Path:
        """Serialise and save a detector to the store directory.

        Both a ``.joblib`` model file and a ``.meta.json`` sidecar are written.

        Args:
            detector: The fitted detector to save.
            pipeline_name: Logical name for the pipeline this detector covers.
                Defaults to ``"global"``.

        Returns:
            Path to the saved ``.joblib`` file.
        """
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = f"{pipeline_name}_{detector.detector_name}_{timestamp}"
        model_path = self._store_dir / f"{stem}.joblib"
        meta_path = self._store_dir / f"{stem}{_METADATA_SUFFIX}"

        joblib.dump(detector, model_path)

        metadata = {
            "pipeline_name": pipeline_name,
            "detector_name": detector.detector_name,
            "fitted_at": timestamp,
            "model_path": str(model_path),
            "threshold": getattr(detector, "threshold", None),
        }
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        log.info(
            "model_store_saved",
            model_path=str(model_path),
            pipeline_name=pipeline_name,
            detector=detector.detector_name,
        )
        return model_path

    def load(self, path: Union[str, Path]) -> BaseDetector:
        """Load a detector from a ``.joblib`` file.

        Args:
            path: Filesystem path to the ``.joblib`` model file.

        Returns:
            The deserialised :class:`~pipeline_anomaly_detector.models.BaseDetector`.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        detector: BaseDetector = joblib.load(path)
        log.info("model_store_loaded", path=str(path))
        return detector

    def load_latest(
        self,
        pipeline_name: str = "global",
        detector_name: str | None = None,
    ) -> BaseDetector:
        """Load the most recently saved detector for a pipeline.

        Args:
            pipeline_name: Pipeline name to filter by.
            detector_name: Optional detector type to filter by (e.g.
                ``"isolation_forest"``).

        Returns:
            The most recently saved :class:`~pipeline_anomaly_detector.models.BaseDetector`.

        Raises:
            FileNotFoundError: If no matching model is found.
        """
        models = self.list_models()
        filtered = [
            m for m in models if m["pipeline_name"] == pipeline_name
        ]
        if detector_name is not None:
            filtered = [m for m in filtered if m["detector_name"] == detector_name]

        if not filtered:
            raise FileNotFoundError(
                f"No model found for pipeline_name='{pipeline_name}'"
                + (f", detector_name='{detector_name}'" if detector_name else "")
            )

        # list_models returns sorted desc by fitted_at, so first is latest
        latest = filtered[0]
        return self.load(latest["model_path"])

    def list_models(self) -> list[dict]:
        """Return metadata for all saved models, newest first.

        Returns:
            List of metadata dicts sorted by ``fitted_at`` descending.
        """
        meta_files = sorted(self._store_dir.glob(f"*{_METADATA_SUFFIX}"))
        results: list[dict] = []
        for meta_path in meta_files:
            try:
                with meta_path.open("r", encoding="utf-8") as fh:
                    metadata = json.load(fh)
                results.append(metadata)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "model_store_meta_read_error",
                    path=str(meta_path),
                    error=str(exc),
                )

        results.sort(key=lambda m: m.get("fitted_at", ""), reverse=True)
        return results
