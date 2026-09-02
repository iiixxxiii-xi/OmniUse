"""The ``ablation`` command: baseline (recovery off) vs full (recovery on)."""

import asyncio
from pathlib import Path

import typer

from minicua.cli.common import load_script, resolve_model
from minicua.controller.llm import ChatModel, FakeModel
from minicua.eval.ablation import AblationResult, render_ablation_markdown, run_ablation
from minicua.eval.errors import TaskDefinitionError
from minicua.eval.metrics_aggregate import aggregate
from minicua.eval.runner import SuiteResult, run_task
from minicua.eval.task import load_tasks


def _run_scripted_ablation(tasks, responses: list, max_steps: int) -> AblationResult:
    """Run each task in both modes with a *fresh* :class:`FakeModel` per task+mode."""
    baseline_results = [
        asyncio.run(
            run_task(task, FakeModel(responses=list(responses)), max_steps=max_steps, recovery=False)
        )
        for task in tasks
    ]
    full_results = [
        asyncio.run(
            run_task(task, FakeModel(responses=list(responses)), max_steps=max_steps, recovery=True)
        )
        for task in tasks
    ]
    baseline = SuiteResult(
        results=baseline_results,
        metrics=aggregate([r.event_log for r in baseline_results], [r.score for r in baseline_results]),
    )
    full = SuiteResult(
        results=full_results,
        metrics=aggregate([r.event_log for r in full_results], [r.score for r in full_results]),
    )
    return AblationResult(baseline=baseline, full=full)


def ablation_command(
    tasks_dir: str = typer.Argument(..., help="Directory (or file) of task JSONs."),
    output: str = typer.Option(
        ".",
        "--output",
        "-o",
        help="Directory to write ablation.md / ablation.json.",
    ),
    script: str | None = typer.Option(
        None,
        "--script",
        help="Optional JSON list of scripted responses applied to each task.",
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
    max_steps: int = typer.Option(20, "--max-steps", help="Maximum agent steps per task."),
) -> None:
    """Run a task set as a bare ReAct loop and with full recovery, then compare."""
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
        ablation = _run_scripted_ablation(tasks, responses, max_steps)
    else:
        model_obj: ChatModel
        try:
            model_obj = resolve_model(model)
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2)
        ablation = asyncio.run(
            run_ablation(tasks, lambda: model_obj, max_steps=max_steps, use_vision=use_vision)
        )

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.md").write_text(render_ablation_markdown(ablation), encoding="utf-8")
    (out / "ablation.json").write_text(ablation.model_dump_json(indent=2), encoding="utf-8")

    comparison = ablation.comparison()
    typer.echo(
        f"success rate: baseline {comparison['baseline_success_rate'] * 100:.1f}% -> "
        f"full {comparison['full_success_rate'] * 100:.1f}% "
        f"(delta {comparison['success_rate_delta'] * 100:+.1f}%)"
    )
    typer.echo(
        f"invalid action: baseline {comparison['baseline_invalid_action_rate'] * 100:.1f}% -> "
        f"full {comparison['full_invalid_action_rate'] * 100:.1f}%"
    )
    typer.echo(f"recovery success: {comparison['full_recovery_success_rate'] * 100:.1f}%")
    typer.echo(f"wrote ablation report to {out}")
    raise typer.Exit(0)
