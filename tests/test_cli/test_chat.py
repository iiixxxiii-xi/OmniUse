"""The ``chat`` CLI command: argument wiring + the interactive REPL's exit logic."""

from typer.testing import CliRunner

from minicua.chat import ChatRun
from minicua.cli import chat as chat_mod
from minicua.cli.main import app
from minicua.controller.llm import FakeModel

runner = CliRunner()


# --------------------------------------------------------------------------- #
# CLI argument wiring
# --------------------------------------------------------------------------- #


def test_chat_help():
    result = runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--use-vision" in result.output
    assert "--max-steps" in result.output
    assert "--headless" in result.output


def test_chat_wires_model_vision_steps(monkeypatch):
    captured = {}

    class _StubRunner:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs

    monkeypatch.setattr(chat_mod, "resolve_model", lambda mid: FakeModel())
    monkeypatch.setattr(chat_mod, "ChatRunner", _StubRunner)
    monkeypatch.setattr(chat_mod, "run_repl", lambda runner: captured.setdefault("repl", runner))

    result = runner.invoke(
        app,
        ["chat", "--model", "dashscope/qwen3-vl-flash", "--use-vision", "vision", "--max-steps", "7"],
    )
    assert result.exit_code == 0, result.output
    assert isinstance(captured["model"], FakeModel)
    assert captured["kwargs"]["use_vision"] == "vision"
    assert captured["kwargs"]["max_steps"] == 7
    assert captured["kwargs"]["headless"] is False


def test_chat_model_error_exits_nonzero(monkeypatch):
    def _bad_model(mid):
        raise ValueError("unrecognized model")

    monkeypatch.setattr(chat_mod, "resolve_model", _bad_model)
    result = runner.invoke(app, ["chat", "--model", "bogus/x"])
    assert result.exit_code == 2
    assert "error" in result.output.lower()


# --------------------------------------------------------------------------- #
# REPL exit logic (tested directly, no browser / no typer)
# --------------------------------------------------------------------------- #


class _StubRunner:
    """An object with an async ``run`` that returns a canned ChatRun (no browser)."""

    def __init__(self):
        self.instructions = []

    async def run(self, instruction, **kwargs):
        self.instructions.append(instruction)
        if instruction == "boom":
            raise RuntimeError("kaboom")
        return ChatRun(instruction=instruction, final_url="http://x/", summary="1. finished")


def _input_fn(*lines):
    it = iter(lines)

    def _read(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    return _read


def test_repl_runs_instruction_then_exit():
    stub = _StubRunner()
    out = []
    chat_mod.run_repl(stub, input_fn=_input_fn("click it", "exit"), echo=out.append)

    assert stub.instructions == ["click it"]
    assert any("final URL: http://x/" in line for line in out)


def test_repl_skips_empty_and_whitespace():
    stub = _StubRunner()
    chat_mod.run_repl(stub, input_fn=_input_fn("", "   ", "do thing", "quit"), echo=lambda s: None)

    assert stub.instructions == ["do thing"]


def test_repl_exit_and_quit_run_nothing():
    for word in ("exit", "quit"):
        stub = _StubRunner()
        chat_mod.run_repl(stub, input_fn=_input_fn(word), echo=lambda s: None)
        assert stub.instructions == []


def test_repl_eof_exits_cleanly():
    stub = _StubRunner()
    chat_mod.run_repl(stub, input_fn=_input_fn(), echo=lambda s: None)
    assert stub.instructions == []


def test_repl_keyboard_interrupt_exits_cleanly():
    stub = _StubRunner()

    def _interrupt(prompt=""):
        raise KeyboardInterrupt

    chat_mod.run_repl(stub, input_fn=_interrupt, echo=lambda s: None)
    assert stub.instructions == []


def test_repl_catches_run_errors_and_continues():
    stub = _StubRunner()
    out = []
    chat_mod.run_repl(stub, input_fn=_input_fn("boom", "ok", "exit"), echo=out.append)

    assert stub.instructions == ["boom", "ok"]
    assert any("error: kaboom" in line for line in out)
