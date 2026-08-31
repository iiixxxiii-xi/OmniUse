"""The ``chat`` command: an interactive conversational browser REPL.

Each line of natural language the user types is fed to the agent and run in a
fresh :class:`BrowserSession` (via :class:`~minicua.chat.runner.ChatRunner`), then
the actions taken, the final URL, and a short summary are printed. There is no
task JSON and no evaluator — the human watches the browser / output.
"""

import asyncio
from collections.abc import Callable

import typer

from minicua.chat.runner import ChatRun, ChatRunner
from minicua.cli.common import resolve_model

_EXIT_COMMANDS = frozenset({"exit", "quit"})


def format_chat_run(result: ChatRun) -> str:
    """Render a :class:`ChatRun` for the terminal: final URL + summary + a note."""
    lines = [f"final URL: {result.final_url}"]
    if result.summary:
        lines.append(result.summary)
    if result.error:
        lines.append(f"note: stopped ({result.stop_reason}): {result.error}")
    elif result.stop_reason and result.stop_reason != "done":
        lines.append(f"note: stopped ({result.stop_reason})")
    return "\n".join(lines)


def run_repl(
    chat_runner: ChatRunner,
    *,
    input_fn: Callable[[str], str] = input,
    echo: Callable[[str], None] = typer.echo,
) -> None:
    """Run the interactive loop until ``exit``/``quit``, EOF, or Ctrl+C.

    ``input_fn`` / ``echo`` are injectable so the exit logic is testable without
    a real TTY or browser. Ctrl+C (or any per-instruction error) never leaves a
    browser process behind: :meth:`ChatRunner.run` closes its session in a
    ``finally``.
    """
    echo("minicua chat — type a browser instruction (exit/quit to quit, Ctrl+C to abort).")
    while True:
        try:
            line = input_fn("> ")
        except KeyboardInterrupt:
            echo("\nBye.")
            return
        except EOFError:
            echo("")
            return

        line = line.strip()
        if not line:
            continue
        if line.lower() in _EXIT_COMMANDS:
            echo("Bye.")
            return

        try:
            result = asyncio.run(chat_runner.run(line))
        except KeyboardInterrupt:
            echo("\nInterrupted.")
            return
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive on any error
            echo(f"error: {exc}")
            continue

        echo(format_chat_run(result))


def chat_command(
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
    max_steps: int = typer.Option(20, "--max-steps", help="Maximum agent steps per instruction."),
    headless: bool = typer.Option(
        False,
        "--headless",
        help="Run headless (hide the browser window; default shows it so you can watch).",
    ),
) -> None:
    """Start an interactive REPL that drives a browser from natural-language instructions."""
    try:
        model_obj = resolve_model(model)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2)

    runner = ChatRunner(model_obj, max_steps=max_steps, use_vision=use_vision, headless=headless)
    run_repl(runner)
    raise typer.Exit(0)
