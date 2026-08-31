"""The minicua CLI: ``run`` (one task), ``eval`` (a task set), ``report`` (re-render)."""

import typer

from minicua.cli.chat import chat_command
from minicua.cli.eval import eval_command
from minicua.cli.report import report_command
from minicua.cli.run import run_command

app = typer.Typer(
    name="minicua",
    help="Long-horizon browser-first computer-use agent (eval + CLI).",
    no_args_is_help=True,
)

app.command("run")(run_command)
app.command("eval")(eval_command)
app.command("report")(report_command)
app.command("chat")(chat_command)


def main() -> None:
    """Entry point for ``python -m minicua`` and the ``minicua`` console script."""
    app()


if __name__ == "__main__":
    main()
