"""The ``eval`` command: run a whole task set and write reports."""

import asyncio
from pathlib import Path

import typer

from minicua.cli.common import load_script
from minicua.controller.llm import FakeModel
from minicua.eval.errors import TaskDefinitionError
from minicua.eval.metrics_aggregate import aggregate
from minicua.eval.report import write_report
from minicua.eval.runner import SuiteResult, run_suite, run_task
from minicua.eval.task import TaskDef, load_tasks


def _run_scripted_suite(
    tasks: list[TaskDef],
    responses: list,
    max_steps: int,
) -> SuiteResult:
    """Run each task with a *fresh* :class:`FakeModel` seeded from the same script."""
    results = []
    for task in tasks:
        model = FakeModel(responses=list(responses))
        results.append(asyncio.run(run_task(task, model, max_steps=max_steps)))
    metrics = aggregate([r.event_log for r in results], [r.score for r in results])
    return SuiteResult(results=results, metrics=metrics)


def eval_command(
    tasks_dir: str = typer.Argument(..., help="Directory (or file) of task JSONs."),
    output: str = typer.Option(
        ".",
        "--output",
        "-o",
        help="Directory to write report.md / report.csv / results.json.",
    ),
    script: str | None = typer.Option(
        None,
        "--script",
        help="Optional JSON list of scripted responses applied to each task.",
    ),
    max_steps: int = typer.Option(20, "--max-steps", help="Maximum agent steps per task."),
) -> None:
    """Run a task set and write markdown + CSV + JSON reports."""
    try:
        tasks = load_tasks(tasks_dir)
    except TaskDefinitionError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)
    if not tasks:
        typer.echo("error: no tasks found", err=True)
        raise typer.Exit(2)

    if script:
        try:
            responses = load_script(script)
        except (OSError, ValueError) as exc:
            typer.echo(f"error: cannot load script: {exc}", err=True)
            raise typer.Exit(2)
        suite = _run_scripted_suite(tasks, responses, max_steps)
    else:
        suite = asyncio.run(run_suite(tasks, FakeModel(), max_steps=max_steps))

    out = Path(output)
    write_report(suite, out)
    (out / "results.json").write_text(suite.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"{suite.n_passed}/{suite.n_total} tasks passed ({suite.success_rate * 100:.1f}%)")
    typer.echo(f"wrote reports to {out}")
    raise typer.Exit(0)
