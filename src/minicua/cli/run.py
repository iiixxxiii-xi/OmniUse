"""The ``run`` command: run a single browser task and print its outcome."""

import asyncio

import typer

from minicua.cli.common import load_script
from minicua.controller.llm import FakeModel
from minicua.eval.errors import TaskDefinitionError
from minicua.eval.runner import run_task
from minicua.eval.task import load_task_file


def run_command(
    task_file: str = typer.Argument(..., help="Path to a task JSON file."),
    script: str | None = typer.Option(
        None,
        "--script",
        help="Path to a JSON list of scripted FakeModel responses (no API key needed).",
    ),
    max_steps: int = typer.Option(20, "--max-steps", help="Maximum agent steps."),
) -> None:
    """Run a single browser task and print its outcome.

    Uses a FakeModel by default (no API key), driven by an optional --script.
    Exit code 0 means the task passed its declarative evaluator.
    """
    try:
        task = load_task_file(task_file)
    except TaskDefinitionError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)

    try:
        responses = load_script(script) if script else []
    except (OSError, ValueError) as exc:
        typer.echo(f"error: cannot load script: {exc}", err=True)
        raise typer.Exit(2)

    result = asyncio.run(run_task(task, FakeModel(responses=responses), max_steps=max_steps))
    typer.echo(f"task: {result.task_id}")
    typer.echo(
        f"success: {result.success}  score: {result.score:.3f}  "
        f"steps: {result.steps}  stop_reason: {result.stop_reason}"
    )
    if result.error:
        typer.echo(f"error: {result.error}")
    raise typer.Exit(0 if result.success else 1)
