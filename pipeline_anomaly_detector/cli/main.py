"""Command-line interface for the Pipeline Anomaly Detector."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

log = structlog.get_logger(__name__)

app = typer.Typer(
    name="pad",
    help="Pipeline Anomaly Detector — detect anomalies in your data pipelines.",
    add_completion=False,
)
console = Console()

# Sub-app for model management commands
models_app = typer.Typer(help="Manage trained detector models.")
app.add_typer(models_app, name="models")


# ---------------------------------------------------------------------------
# Helper: parse --since DATE string
# ---------------------------------------------------------------------------

def _parse_since(since_str: str | None) -> datetime:
    """Parse a date string into a UTC-aware datetime.

    Args:
        since_str: ISO date/datetime string (e.g. ``"2024-01-01"``).

    Returns:
        UTC-aware :class:`datetime`.
    """
    if since_str is None:
        # Default: 30 days ago
        from datetime import timedelta
        return datetime.now(tz=timezone.utc) - timedelta(days=30)
    try:
        dt = datetime.fromisoformat(since_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        console.print(f"[red]Invalid date format: {since_str!r}. Use ISO format, e.g. 2024-01-01[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

@app.command("collect")
def collect_cmd(
    source: str = typer.Option(..., help="Source type: dbt | airflow | generic"),
    since: Optional[str] = typer.Option(None, help="Lower bound date (ISO format). Default: 30 days ago."),
    output: str = typer.Option(..., help="Output JSONL file path."),
    dbt_dir: Optional[str] = typer.Option(None, help="Glob pattern for dbt run_results.json files."),
    airflow_db: Optional[str] = typer.Option(None, help="Airflow metadata DB SQLAlchemy URL."),
    input_file: Optional[str] = typer.Option(None, "--input", help="Input JSON/JSONL file for generic source."),
) -> None:
    """Collect pipeline runs from a source and write them to a JSONL file.

    Examples::

        pad collect --source dbt --dbt-dir "./target/run_results.json" --since 2024-01-01 --output runs.jsonl
        pad collect --source generic --input runs.json --output out.jsonl
        pad collect --source airflow --airflow-db "sqlite:///airflow.db" --output runs.jsonl
    """
    since_dt = _parse_since(since)

    if source == "dbt":
        if not dbt_dir:
            console.print("[red]--dbt-dir is required for source=dbt[/red]")
            raise typer.Exit(code=1)
        from pipeline_anomaly_detector.collectors.dbt_collector import DbtCollector
        collector = DbtCollector(directory_glob=dbt_dir)

    elif source == "airflow":
        if not airflow_db:
            console.print("[red]--airflow-db is required for source=airflow[/red]")
            raise typer.Exit(code=1)
        from pipeline_anomaly_detector.collectors.airflow_collector import AirflowCollector
        collector = AirflowCollector(db_url=airflow_db)

    elif source == "generic":
        src = input_file or "runs.jsonl"
        from pipeline_anomaly_detector.collectors.generic_collector import GenericCollector
        collector = GenericCollector(source=src)

    else:
        console.print(f"[red]Unknown source: {source!r}. Choose from: dbt, airflow, generic[/red]")
        raise typer.Exit(code=1)

    with console.status(f"Collecting from [bold]{source}[/bold]..."):
        runs = collector.collect(since=since_dt)

    output_path = Path(output)
    with output_path.open("w", encoding="utf-8") as fh:
        for run in runs:
            fh.write(run.model_dump_json() + "\n")

    console.print(
        f"[green]Collected {len(runs)} runs → {output_path}[/green]"
    )


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@app.command("train")
def train_cmd(
    input_file: str = typer.Option(..., "--input", help="Input JSONL file with pipeline runs."),
    detector: str = typer.Option("ensemble", help="Detector type: isolation_forest | zscore | ensemble"),
    output_dir: str = typer.Option(..., "--output", help="Directory to save the trained model."),
    since: Optional[str] = typer.Option(None, help="Only train on runs after this date (ISO format)."),
) -> None:
    """Train an anomaly detector on collected pipeline runs.

    Example::

        pad train --input runs.jsonl --detector ensemble --output ./models
    """
    from pipeline_anomaly_detector.collectors.generic_collector import GenericCollector
    from pipeline_anomaly_detector.training.trainer import Trainer
    from pipeline_anomaly_detector.training.model_store import ModelStore

    since_dt = _parse_since(since)
    collector = GenericCollector(source=input_file)

    if detector == "isolation_forest":
        from pipeline_anomaly_detector.models.isolation_forest_detector import IsolationForestDetector
        det = IsolationForestDetector()
    elif detector == "zscore":
        from pipeline_anomaly_detector.models.zscore_detector import ZScoreDetector
        det = ZScoreDetector()
    elif detector == "ensemble":
        from pipeline_anomaly_detector.models.isolation_forest_detector import IsolationForestDetector
        from pipeline_anomaly_detector.models.zscore_detector import ZScoreDetector
        from pipeline_anomaly_detector.models.ensemble_detector import EnsembleDetector
        det = EnsembleDetector(detectors=[ZScoreDetector(), IsolationForestDetector()])
    else:
        console.print(f"[red]Unknown detector: {detector!r}[/red]")
        raise typer.Exit(code=1)

    trainer = Trainer()
    with console.status(f"Training [bold]{detector}[/bold] detector..."):
        result = trainer.fit(
            collector=collector,
            since=since_dt,
            detector=det,
        )

    store = ModelStore(store_dir=output_dir)
    model_path = store.save(result.detector)

    stats = result.training_stats
    table = Table(title="Training Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Detector", detector)
    table.add_row("Training Runs", str(stats.get("n_runs", 0)))
    table.add_row("Pipelines", str(stats.get("n_pipelines", 0)))
    table.add_row("Model Path", str(model_path))
    console.print(table)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

@app.command("score")
def score_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Run ID to look up and score."),
    model: str = typer.Option(..., help="Path to the .joblib model file."),
    db: str = typer.Option("anomaly_scores.db", help="Path to the SQLite scores database."),
) -> None:
    """Score a single run by ID.

    The run must exist in the scores database. Use ``pad explain`` for a
    detailed breakdown.

    Example::

        pad score --run-id run_001 --model ./models/model.joblib --db scores.db
    """
    from pipeline_anomaly_detector.training.model_store import ModelStore
    from pipeline_anomaly_detector.scoring.scorer import Scorer

    det = ModelStore(store_dir=str(Path(model).parent)).load(model)
    scorer = Scorer(detector=det, db_path=db)
    history = scorer.get_history(pipeline_name="", limit=1000)

    # Search for the run_id in history
    match = next((h for h in history if h["run_id"] == run_id), None)
    if match is None:
        console.print(f"[yellow]Run ID '{run_id}' not found in database {db!r}.[/yellow]")
        console.print("Use 'pad score-batch' to score runs from a file first.")
        raise typer.Exit(code=1)

    _print_score_result(match)


# ---------------------------------------------------------------------------
# score-batch
# ---------------------------------------------------------------------------

@app.command("score-batch")
def score_batch_cmd(
    input_file: str = typer.Option(..., "--input", help="JSONL file of pipeline runs to score."),
    model: str = typer.Option(..., help="Path to the .joblib model file."),
    db: str = typer.Option("anomaly_scores.db", help="SQLite database path."),
) -> None:
    """Score a batch of pipeline runs from a JSONL file.

    Example::

        pad score-batch --input runs.jsonl --model ./models/model.joblib --db scores.db
    """
    from pipeline_anomaly_detector.collectors.generic_collector import GenericCollector
    from pipeline_anomaly_detector.training.model_store import ModelStore
    from pipeline_anomaly_detector.scoring.scorer import Scorer
    from datetime import timezone

    collector = GenericCollector(source=input_file)
    runs = collector.collect(since=datetime.min.replace(tzinfo=timezone.utc))

    det = ModelStore(store_dir=str(Path(model).parent)).load(model)
    scorer = Scorer(detector=det, db_path=db)

    with console.status(f"Scoring {len(runs)} runs..."):
        scores = scorer.score_batch(runs)

    n_anomalies = sum(1 for s in scores if s.is_anomaly)
    table = Table(title=f"Scored {len(scores)} runs — {n_anomalies} anomalies detected")
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Pipeline", style="blue")
    table.add_column("Score", justify="right")
    table.add_column("Anomaly", justify="center")

    for s in scores:
        anomaly_cell = "[red]YES[/red]" if s.is_anomaly else "[green]no[/green]"
        table.add_row(
            s.run_id,
            s.pipeline_name,
            f"{s.anomaly_score:.3f}",
            anomaly_cell,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------

@app.command("explain")
def explain_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Run ID to explain."),
    model: str = typer.Option(..., help="Path to the .joblib model file."),
    db: str = typer.Option("anomaly_scores.db", help="SQLite database path."),
) -> None:
    """Explain an anomaly score with a detailed Rich panel.

    Prints contributing features, anomaly score bar, and is_anomaly status.

    Example::

        pad explain --run-id run_001 --model ./models/model.joblib --db scores.db
    """
    from pipeline_anomaly_detector.training.model_store import ModelStore
    from pipeline_anomaly_detector.scoring.scorer import Scorer

    det = ModelStore(store_dir=str(Path(model).parent)).load(model)
    scorer = Scorer(detector=det, db_path=db)
    history = scorer.get_history(pipeline_name="", limit=10000)
    match = next((h for h in history if h["run_id"] == run_id), None)

    if match is None:
        console.print(f"[yellow]Run ID '{run_id}' not found in {db!r}[/yellow]")
        raise typer.Exit(code=1)

    _print_explain_panel(match)


# ---------------------------------------------------------------------------
# models list
# ---------------------------------------------------------------------------

@models_app.command("list")
def models_list_cmd(
    store_dir: str = typer.Option("./models", help="Model store directory."),
) -> None:
    """List all saved models in the model store.

    Example::

        pad models list --store-dir ./models
    """
    from pipeline_anomaly_detector.training.model_store import ModelStore

    store = ModelStore(store_dir=store_dir)
    models = store.list_models()

    if not models:
        console.print(f"[yellow]No models found in {store_dir!r}[/yellow]")
        return

    table = Table(title=f"Models in {store_dir}")
    table.add_column("Pipeline", style="blue")
    table.add_column("Detector", style="cyan")
    table.add_column("Fitted At", style="green")
    table.add_column("Path", style="dim")

    for m in models:
        table.add_row(
            m.get("pipeline_name", ""),
            m.get("detector_name", ""),
            m.get("fitted_at", ""),
            m.get("model_path", ""),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Internal render helpers
# ---------------------------------------------------------------------------

def _print_score_result(match: dict) -> None:
    """Print a brief score result summary.

    Args:
        match: Score dict from scorer history.
    """
    score = match["anomaly_score"]
    is_anomaly = match["is_anomaly"]
    status = "[red]ANOMALY[/red]" if is_anomaly else "[green]normal[/green]"
    console.print(f"Run ID: [bold]{match['run_id']}[/bold]")
    console.print(f"Pipeline: {match['pipeline_name']}")
    console.print(f"Score: {score:.4f}  Status: {status}")
    console.print(f"Features: {match.get('contributing_features', [])}")


def _print_explain_panel(match: dict) -> None:
    """Print a rich explanation panel for an anomaly score.

    Args:
        match: Score dict from scorer history.
    """
    score = match["anomaly_score"]
    is_anomaly = match["is_anomaly"]
    pct = int(score * 100)
    filled = int(round(score * 20))
    bar = "█" * filled + "░" * (20 - filled)

    status_txt = Text("ANOMALY DETECTED", style="bold red") if is_anomaly else Text("Normal", style="bold green")
    features = match.get("contributing_features", [])
    feature_lines = "\n".join(f"  • {f}" for f in features) if features else "  (none)"

    content = (
        f"Run ID:   {match['run_id']}\n"
        f"Pipeline: {match['pipeline_name']}\n"
        f"Detector: {match.get('detector_name', '')}\n"
        f"Time:     {match.get('timestamp', '')}\n\n"
        f"Score: [{bar}] {pct}%\n\n"
        f"Contributing Features:\n{feature_lines}"
    )

    title = Text(f"  {'🚨' if is_anomaly else '✅'} {status_txt}  ", style="bold")
    panel = Panel(
        content,
        title=str(title),
        border_style="red" if is_anomaly else "green",
        expand=False,
    )
    console.print(panel)


if __name__ == "__main__":
    app()
