"""CLI ``--mode desktop`` wiring for the ``run`` and ``chat`` commands."""

import json

from typer.testing import CliRunner

from minicua.cli import run as run_mod
from minicua.cli.main import app
from minicua.controller.llm import FakeModel

runner = CliRunner()

_DESKTOP_TASK = {"id": "dt1", "instruction": "open notepad"}


def test_run_help_shows_mode_option():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--mode" in result.output


def test_chat_help_shows_mode_option():
    result = runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--mode" in result.output


def test_chat_desktop_mode_rejects_non_vision_model():
    # Default model is "fake" (text-only); desktop mode must reject it.
    result = runner.invoke(app, ["chat", "--mode", "desktop"], input="exit\n")
    assert result.exit_code == 2
    assert "vision" in result.output.lower()


def test_run_desktop_mode_runs(tmp_path, monkeypatch):
    task_file = tmp_path / "task.json"
    task_file.write_text(json.dumps(_DESKTOP_TASK), encoding="utf-8")

    captured = {}

    async def fake_run_desktop(instruction, model, *, environment=None, max_steps=20, use_vision="vision"):
        captured["instruction"] = instruction
        captured["max_steps"] = max_steps
        from minicua.desktop.runner import DesktopRunResult

        return DesktopRunResult(task_id="dt1", success=True, steps=1, stop_reason="done")

    monkeypatch.setattr(run_mod, "resolve_model", lambda mid: FakeModel(supports_vision=True))
    monkeypatch.setattr(run_mod, "run_desktop", fake_run_desktop)

    result = runner.invoke(
        app, ["run", str(task_file), "--mode", "desktop", "--model", "dashscope/qwen3-vl-flash"]
    )
    assert result.exit_code == 0, result.output
    assert captured["instruction"] == "open notepad"
    assert "success" in result.output.lower()


def test_run_desktop_mode_rejects_non_vision_model(tmp_path, monkeypatch):
    task_file = tmp_path / "task.json"
    task_file.write_text(json.dumps(_DESKTOP_TASK), encoding="utf-8")

    monkeypatch.setattr(run_mod, "resolve_model", lambda mid: FakeModel(supports_vision=False))
    result = runner.invoke(app, ["run", str(task_file), "--mode", "desktop", "--model", "deepseek/x"])
    assert result.exit_code == 2
    assert "vision" in result.output.lower()
