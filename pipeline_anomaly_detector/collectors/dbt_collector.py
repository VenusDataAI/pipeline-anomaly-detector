"""Collector for dbt run_results.json files."""

from __future__ import annotations

import glob as _glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import structlog

from pipeline_anomaly_detector import PipelineRun
from pipeline_anomaly_detector.collectors.base_collector import BaseCollector

log = structlog.get_logger(__name__)

# Mapping from dbt status strings to PipelineRun status literals.
_STATUS_MAP: dict[str, str] = {
    "success": "success",
    "pass": "success",
    "warn": "success",
    "error": "failed",
    "fail": "failed",
    "skipped": "skipped",
    "runtime error": "failed",
}


class DbtCollector(BaseCollector):
    """Collect pipeline runs by parsing dbt ``run_results.json`` files.

    The collector accepts a glob pattern that matches one or more
    ``run_results.json`` files produced by ``dbt run`` / ``dbt test``.
    Each *node result* inside the file is mapped to a
    :class:`~pipeline_anomaly_detector.PipelineRun`.

    Args:
        directory_glob: A glob pattern (e.g. ``"./target/run_results.json"``
            or ``"/data/projects/**/target/run_results.json"``) used to
            discover result files.

    Example::

        collector = DbtCollector(directory_glob="./target/run_results.json")
        runs = collector.collect(since=datetime(2024, 1, 1, tzinfo=timezone.utc))
    """

    def __init__(self, directory_glob: str) -> None:
        """Initialise with a glob pattern for locating result files.

        Args:
            directory_glob: Shell-style glob pattern for ``run_results.json``
                files (supports ``**`` with :func:`glob.glob`).
        """
        self._glob = directory_glob
        log.debug("dbt_collector_initialised", glob=directory_glob)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_files(self) -> list[Path]:
        """Expand the glob pattern to a sorted list of file paths.

        Returns:
            Sorted list of :class:`~pathlib.Path` objects.
        """
        matched = _glob.glob(self._glob, recursive=True)
        paths = sorted(Path(p) for p in matched if Path(p).is_file())
        log.debug("dbt_collector_files_found", count=len(paths), glob=self._glob)
        return paths

    @staticmethod
    def _parse_file(path: Path) -> list[dict]:
        """Read and parse a single ``run_results.json`` file.

        Args:
            path: Filesystem path to the file.

        Returns:
            List of raw result dicts from the ``results`` key.
        """
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("results", [])

    @staticmethod
    def _extract_duration(result: dict) -> float:
        """Extract execution duration from the timing block or elapsed_time.

        dbt stores a ``timing`` list of ``{"name": "...", "started_at": ...,
        "completed_at": ...}`` blocks.  The *execute* timing block covers the
        actual SQL execution; we fall back to ``execution_time`` /
        ``elapsed_time`` if the timing block is absent.

        Args:
            result: A single result node dict from ``run_results.json``.

        Returns:
            Duration in seconds (float), or 0.0 if unavailable.
        """
        timing: list[dict] = result.get("timing", [])
        for block in timing:
            if block.get("name") == "execute":
                started = block.get("started_at")
                completed = block.get("completed_at")
                if started and completed:
                    try:
                        s = datetime.fromisoformat(started.replace("Z", "+00:00"))
                        c = datetime.fromisoformat(completed.replace("Z", "+00:00"))
                        return max(0.0, (c - s).total_seconds())
                    except (ValueError, TypeError):
                        pass

        # Fallback: sum all timing blocks or use execution_time
        for key in ("execution_time", "elapsed_time"):
            val = result.get(key)
            if isinstance(val, (int, float)):
                return float(val)

        return 0.0

    @staticmethod
    def _extract_rows(result: dict) -> int:
        """Extract row count from adapter_response if present.

        Args:
            result: A single result node dict.

        Returns:
            Number of rows processed, or 0 if unavailable.
        """
        adapter_response = result.get("adapter_response", {}) or {}
        for key in ("rows_affected", "rows_processed", "num_rows_affected"):
            val = adapter_response.get(key)
            if isinstance(val, int):
                return max(0, val)
        return 0

    @staticmethod
    def _extract_start_time(result: dict) -> datetime:
        """Extract the run start time from the timing block.

        Args:
            result: A single result node dict.

        Returns:
            A timezone-aware :class:`~datetime.datetime` in UTC.
        """
        timing: list[dict] = result.get("timing", [])
        # Use the earliest started_at across all timing blocks
        candidates: list[datetime] = []
        for block in timing:
            started = block.get("started_at")
            if started:
                try:
                    dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    candidates.append(dt)
                except (ValueError, TypeError):
                    pass
        if candidates:
            return min(candidates)
        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _map_status(result: dict) -> str:
        """Map a dbt status string to a PipelineRun status literal.

        Args:
            result: A single result node dict.

        Returns:
            One of ``"success"``, ``"failed"``, or ``"skipped"``.
        """
        raw = str(result.get("status", "")).lower()
        return _STATUS_MAP.get(raw, "failed")

    def _result_to_run(self, result: dict, file_path: Path) -> PipelineRun | None:
        """Convert a single dbt result node to a :class:`~pipeline_anomaly_detector.PipelineRun`.

        Args:
            result: Raw result dict from ``run_results.json``.
            file_path: Source file path (used in run_id generation).

        Returns:
            A :class:`~pipeline_anomaly_detector.PipelineRun` or ``None`` if
            the result cannot be mapped (e.g. missing ``unique_id``).
        """
        unique_id: str = result.get("unique_id", "")
        if not unique_id:
            log.warning("dbt_collector_missing_unique_id", result=result)
            return None

        # pipeline_name: use the last part of the unique_id (e.g. "model.project.orders" -> "orders")
        parts = unique_id.split(".")
        pipeline_name = parts[-1] if parts else unique_id

        duration = self._extract_duration(result)
        start_time = self._extract_start_time(result)
        end_time_dt = datetime.fromtimestamp(
            start_time.timestamp() + duration, tz=timezone.utc
        )
        rows = self._extract_rows(result)
        status = self._map_status(result)

        # Build a stable run_id
        run_id = f"dbt_{unique_id}_{int(start_time.timestamp())}"

        try:
            return PipelineRun(
                run_id=run_id,
                pipeline_name=pipeline_name,
                start_time=start_time,
                end_time=end_time_dt,
                duration_seconds=max(duration, 0.0),
                rows_processed=rows,
                rows_rejected=0,
                null_rate={},
                status=status,  # type: ignore[arg-type]
                metadata={
                    "unique_id": unique_id,
                    "source_file": str(file_path),
                    "dbt_status": result.get("status", ""),
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "dbt_collector_run_creation_error",
                unique_id=unique_id,
                error=str(exc),
            )
            return None

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    def collect(self, since: datetime) -> list[PipelineRun]:
        """Collect pipeline runs parsed from dbt result files.

        Args:
            since: Only runs with ``start_time >= since`` are returned.

        Returns:
            List of :class:`~pipeline_anomaly_detector.PipelineRun` objects
            sorted by ``start_time`` ascending.
        """
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)

        files = self._find_files()
        runs: list[PipelineRun] = []

        for file_path in files:
            try:
                results = self._parse_file(file_path)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "dbt_collector_file_parse_error",
                    path=str(file_path),
                    error=str(exc),
                )
                continue

            for result in results:
                run = self._result_to_run(result, file_path)
                if run is None:
                    continue
                run_start = run.start_time
                if run_start.tzinfo is None:
                    run_start = run_start.replace(tzinfo=timezone.utc)
                if run_start >= since:
                    runs.append(run)

        runs.sort(key=lambda r: r.start_time)
        log.info(
            "dbt_collector_collected",
            files=len(files),
            runs=len(runs),
        )
        return runs
