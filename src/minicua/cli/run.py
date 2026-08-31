"""The ``run`` command: run a single browser (or desktop) task and print its outcome."""

import asyncio
import json
from pathlib import Path

import typer

from minicua.cli.common import VALID_MODES, load_script, require_vision_model, resolve_model
from minicua.controller.llm import ChatModel, FakeModel
from minicua.desktop.runner import run_desktop
from minicua.eval.errors import TaskDefinitionError
from minicua.eval.runner import run_task
from minicua.eval.task import load_task_file


def _load_desktop_task(path: str | Path) -> tuple[str, str]:
    """Load ``(id, instruction)`` from a desktop task JSON (no evaluator/html needed)."""
    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskDefinitionError(f"could not read task file {target}: {exc}") from exc
    if not isinstance(data, dict):
        raise TaskDefinitionError(f"task file {target} must contain a JSON object")
    instruction = data.get("instruction")
    if not instruction or not str(instruction).strip():
        raise TaskDefinitionError(f"desktop task file {target} must contain an 'instruction'")
    return str(data.get("id", "")), str(instruction)


def run_command(
    task_file: str = typer.Argument(..., help="Path to a task JSON file."),
    script: str | None = typer.Option(
        None,
        "--script",
        help="Path to a JSON list of scripted FakeModel responses (no API key needed).",
    ),
    model: str = typer.Option(
        "fake",
        "--model",
        help="Model: fake, deepseek/<id>, dashscope/<id>, or qwen/<id>.",
    ),
    use_vision: str = typer.Option(
        "auto",
        "--use-vision",
        help="Vision mode: dom_only, vision, or auto (capture iff the model supports it).",
    ),
    max_steps: int = typer.Option(20, "--max-steps", help="Maximum agent steps."),
    mode: str = typer.Option(
        "browser",
        "--mode",
        help="Operating mode: browser (DOM) or desktop (screenshot + mouse/keyboard/shell).",
    ),
) -> None:
    """Run a single task and print its outcome.

    Browser mode uses a FakeModel by default (no API key), driven by an optional
    --script, and scores with the declarative evaluator. Desktop mode drives the
    whole machine (screenshot perception + coordinate/shell actions), requires a
    vision model, and has no DOM evaluator. Exit code 0 means success.
    """
    if mode not in VALID_MODES:
        typer.echo(f"error: mode must be one of {VALID_MODES}", err=True)
        raise typer.Exit(2)

    model_obj: ChatModel
    if script:
        try:
            responses = load_script(script)
        except (OSError, ValueError) as exc:
            typer.echo(f"error: cannot load script: {exc}", err=True)
            raise typer.Exit(2)
        model_obj = FakeModel(responses=responses, supports_vision=(mode == "desktop"))
    else:
        try:
            model_obj = resolve_model(model)
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2)

    try:
        require_vision_model(model_obj, mode)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)

    if mode == "desktop":
        try:
            task_id, instruction = _load_desktop_task(task_file)
        except TaskDefinitionError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2)
        result = asyncio.run(run_desktop(instruction, model_obj, max_steps=max_steps))
        typer.echo(f"task: {task_id or '(desktop)'}")
        typer.echo(f"success: {result.success}  steps: {result.steps}  stop_reason: {result.stop_reason}")
        if result.submission:
            typer.echo(f"submission: {result.submission}")
        if result.error:
            typer.echo(f"error: {result.error}")
        raise typer.Exit(0 if result.success else 1)

    try:
        task = load_task_file(task_file)
    except TaskDefinitionError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)

    result = asyncio.run(run_task(task, model_obj, max_steps=max_steps, use_vision=use_vision))
    typer.echo(f"task: {result.task_id}")
    typer.echo(
        f"success: {result.success}  score: {result.score:.3f}  "
        f"steps: {result.steps}  stop_reason: {result.stop_reason}"
    )
    if result.error:
        typer.echo(f"error: {result.error}")
    raise typer.Exit(0 if result.success else 1)
