"""The ``report`` command: re-render reports from a saved ``results.json``."""

from pathlib import Path

import typer
from pydantic import ValidationError

from minicua.eval.report import write_report
from minicua.eval.runner import SuiteResult


def report_command(
    results_file: str = typer.Argument(..., help="Path to a results.json produced by `eval`."),
    output: str = typer.Option(
        ".",
        "--output",
        "-o",
        help="Directory to write report.md / report.csv.",
    ),
) -> None:
    """Re-render markdown + CSV reports from a saved results.json (no browser needed)."""
    try:
        suite = SuiteResult.model_validate_json(Path(results_file).read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        typer.echo(f"error: cannot load results: {exc}", err=True)
        raise typer.Exit(2)
    md_path, csv_path = write_report(suite, output)
    typer.echo(f"wrote {md_path}")
    typer.echo(f"wrote {csv_path}")
    raise typer.Exit(0)
