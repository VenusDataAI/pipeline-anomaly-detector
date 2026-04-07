"""Feature registry for the pipeline anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata descriptor for a single feature.

    Attributes:
        name: Machine-readable feature name (column name in output DataFrame).
        description: Human-readable description of what the feature captures.
        formula: Textual formula or derivation notes.
        dtype: NumPy/pandas dtype string (e.g. ``"float64"``, ``"int8"``).
        requires_history: Whether this feature needs historical runs to compute.
    """

    name: str
    description: str
    formula: str
    dtype: str
    requires_history: bool


class FeatureRegistry:
    """Registry that tracks all features used by the anomaly detector.

    The registry stores an ordered list of :class:`FeatureSpec` objects and
    exposes helper methods for introspection.

    Example::

        registry = FeatureRegistry()
        df = registry.as_dataframe()
        print(df[["name", "requires_history"]])
    """

    def __init__(self) -> None:
        """Initialise the registry with the canonical 11 features."""
        self._specs: list[FeatureSpec] = []
        self._register_defaults()

    # ------------------------------------------------------------------
    # Internal registration
    # ------------------------------------------------------------------

    def _register_defaults(self) -> None:
        """Register all 11 canonical features."""
        specs = [
            FeatureSpec(
                name="duration_seconds",
                description="Raw wall-clock duration of the pipeline run in seconds.",
                formula="run.duration_seconds",
                dtype="float64",
                requires_history=False,
            ),
            FeatureSpec(
                name="duration_z",
                description=(
                    "Z-score of duration_seconds relative to the rolling 30-run "
                    "per-pipeline window. Captures whether this run is unusually "
                    "slow or fast compared to recent history."
                ),
                formula="(duration_seconds - mean_30) / std_30",
                dtype="float64",
                requires_history=True,
            ),
            FeatureSpec(
                name="rows_processed_log1p",
                description="Natural log1p transform of rows_processed. Reduces skew.",
                formula="log1p(run.rows_processed)",
                dtype="float64",
                requires_history=False,
            ),
            FeatureSpec(
                name="row_count_delta_pct",
                description=(
                    "Percentage change in rows_processed versus the immediately "
                    "preceding run of the same pipeline."
                ),
                formula="(rows_processed - prev_rows_processed) / prev_rows_processed * 100",
                dtype="float64",
                requires_history=True,
            ),
            FeatureSpec(
                name="null_rate_max",
                description="Maximum null rate across all tracked columns for this run.",
                formula="max(run.null_rate.values()) if run.null_rate else 0.0",
                dtype="float64",
                requires_history=False,
            ),
            FeatureSpec(
                name="null_rate_delta",
                description=(
                    "Change in null_rate_max versus the immediately preceding run "
                    "of the same pipeline."
                ),
                formula="null_rate_max - prev_null_rate_max",
                dtype="float64",
                requires_history=True,
            ),
            FeatureSpec(
                name="hour_of_day",
                description="Hour of day (0-23) when the run started (UTC).",
                formula="run.start_time.hour",
                dtype="int8",
                requires_history=False,
            ),
            FeatureSpec(
                name="day_of_week",
                description="Day of week (0=Monday, 6=Sunday) when the run started.",
                formula="run.start_time.weekday()",
                dtype="int8",
                requires_history=False,
            ),
            FeatureSpec(
                name="is_weekend",
                description="Binary flag: 1 if the run started on Saturday or Sunday.",
                formula="1 if run.start_time.weekday() >= 5 else 0",
                dtype="int8",
                requires_history=False,
            ),
            FeatureSpec(
                name="status_is_success",
                description="Binary flag: 1 if run.status == 'success', else 0.",
                formula="1 if run.status == 'success' else 0",
                dtype="int8",
                requires_history=False,
            ),
            FeatureSpec(
                name="rejection_rate",
                description=(
                    "Fraction of rows that were rejected: "
                    "rows_rejected / rows_processed. Zero when rows_processed == 0."
                ),
                formula="rows_rejected / rows_processed if rows_processed > 0 else 0.0",
                dtype="float64",
                requires_history=False,
            ),
        ]
        for spec in specs:
            self._specs.append(spec)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, spec: FeatureSpec) -> None:
        """Register an additional feature specification.

        Args:
            spec: The :class:`FeatureSpec` to add to the registry.
        """
        self._specs.append(spec)

    @property
    def feature_names(self) -> list[str]:
        """Return the ordered list of feature names.

        Returns:
            List of feature name strings.
        """
        return [s.name for s in self._specs]

    def get(self, name: str) -> FeatureSpec:
        """Look up a feature spec by name.

        Args:
            name: The feature name to look up.

        Returns:
            The matching :class:`FeatureSpec`.

        Raises:
            KeyError: If no feature with the given name is registered.
        """
        for spec in self._specs:
            if spec.name == name:
                return spec
        raise KeyError(f"Feature '{name}' is not registered.")

    def as_dataframe(self) -> pd.DataFrame:
        """Return feature metadata as a pandas DataFrame.

        Returns:
            A DataFrame with columns ``name``, ``description``, ``formula``,
            ``dtype``, and ``requires_history``, indexed by feature name.
        """
        records = [
            {
                "name": s.name,
                "description": s.description,
                "formula": s.formula,
                "dtype": s.dtype,
                "requires_history": s.requires_history,
            }
            for s in self._specs
        ]
        df = pd.DataFrame(records).set_index("name")
        return df

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs)


# Module-level singleton for convenient import.
FEATURE_REGISTRY = FeatureRegistry()
