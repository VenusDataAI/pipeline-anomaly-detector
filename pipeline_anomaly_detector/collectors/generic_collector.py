"""Generic collector that accepts raw dicts or a JSONL file path."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Union

import structlog

from pipeline_anomaly_detector import PipelineRun
from pipeline_anomaly_detector.collectors.base_collector import BaseCollector

log = structlog.get_logger(__name__)


class GenericCollector(BaseCollector):
    """Collector that accepts a list of dicts or a path to a JSONL/JSON file.

    Records are validated against the :class:`~pipeline_anomaly_detector.PipelineRun`
    Pydantic schema. Invalid records are skipped with a warning.

    Args:
        source: Either a list of dicts (each representing a pipeline run) or
            a path to a ``.jsonl`` (newline-delimited JSON) or ``.json``
            (JSON array) file.

    Example::

        collector = GenericCollector(source="runs.jsonl")
        runs = collector.collect(since=datetime(2024, 1, 1, tzinfo=timezone.utc))
    """

    def __init__(self, source: Union[list[dict], str, Path]) -> None:
        """Initialise the collector.

        Args:
            source: Raw records as a list of dicts **or** a filesystem path to
                a ``.json`` / ``.jsonl`` file.
        """
        if isinstance(source, (str, Path)):
            self._records = self._load_file(Path(source))
        else:
            self._records = list(source)
        log.debug(
            "generic_collector_initialised",
            n_records=len(self._records),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_file(path: Path) -> list[dict]:
        """Load records from a JSON array or JSONL file.

        Args:
            path: Path to the file to load.

        Returns:
            A list of raw dicts parsed from the file.

        Raises:
            ValueError: If the file extension is not ``.json`` or ``.jsonl``.
        """
        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                raise ValueError(
                    f"Expected a JSON array in {path}, got {type(data).__name__}"
                )
            return data
        elif suffix == ".jsonl":
            records: list[dict] = []
            with path.open("r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        log.warning(
                            "generic_collector_jsonl_parse_error",
                            line=lineno,
                            error=str(exc),
                        )
            return records
        else:
            raise ValueError(
                f"Unsupported file extension '{suffix}'. "
                "Expected '.json' or '.jsonl'."
            )

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    def collect(self, since: datetime) -> list[PipelineRun]:
        """Return validated pipeline runs starting at or after *since*.

        Args:
            since: Lower-bound timestamp (inclusive).

        Returns:
            Validated :class:`~pipeline_anomaly_detector.PipelineRun` objects
            with ``start_time >= since``, ordered by ``start_time`` ascending.
        """
        runs: list[PipelineRun] = []
        skipped = 0

        for idx, record in enumerate(self._records):
            try:
                run = PipelineRun.model_validate(record)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "generic_collector_invalid_record",
                    index=idx,
                    error=str(exc),
                )
                skipped += 1
                continue

            # Normalise timezone for comparison
            run_start = run.start_time
            cmp_since = since
            if run_start.tzinfo is None and cmp_since.tzinfo is not None:
                from datetime import timezone
                run_start = run_start.replace(tzinfo=timezone.utc)
            elif run_start.tzinfo is not None and cmp_since.tzinfo is None:
                from datetime import timezone
                cmp_since = cmp_since.replace(tzinfo=timezone.utc)

            if run_start >= cmp_since:
                runs.append(run)

        runs.sort(key=lambda r: r.start_time)

        log.info(
            "generic_collector_collected",
            total_records=len(self._records),
            valid_runs=len(runs),
            skipped=skipped,
        )
        return runs
