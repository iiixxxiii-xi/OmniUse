"""Task 8.1: the ``run`` CLI command — run a single task and print its outcome."""

import json

from typer.testing import CliRunner

from minicua.cli.main import app

runner = CliRunner()

_CLICK_TASK = {
    "id": "t1",
    "instruction": "click the button",
    "html": "<button id=btn onclick=\"document.getElementById('out').textContent='clicked'\">go</button><div id=out></div>",
    "evaluator": {"func": "exact_match", "result": {"getter": "element_text", "selector": "#out"}, "expected": {"expected": "clicked"}},
}


def test_run_help():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "task" in result.output.lower()


def test_run_missing_task_file_errors():
    result = runner.invoke(app, ["run", "nonexistent_task.json"])
    assert result.exit_code != 0
    assert "error" in result.output.lower()


def test_run_scripted_task_succeeds(tmp_path):
    task_file = tmp_path / "task.json"
    task_file.write_text(json.dumps(_CLICK_TASK), encoding="utf-8")
    script_file = tmp_path / "script.json"
    script_file.write_text(
        json.dumps(
            [
                {"name": "click", "params": {"index": 1}},
                {"name": "done", "params": {"success": True}},
            ]
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["run", str(task_file), "--script", str(script_file)])
    assert result.exit_code == 0, result.output
    assert "success" in result.output.lower()
    assert "score" in result.output.lower()


def test_run_scripted_task_failure_exits_nonzero(tmp_path):
    task_file = tmp_path / "task.json"
    task_file.write_text(json.dumps(_CLICK_TASK), encoding="utf-8")
    script_file = tmp_path / "script.json"
    # The script clicks the wrong index (999), so the evaluator scores 0.
    script_file.write_text(
        json.dumps(
            [
                {"name": "click", "params": {"index": 999}},
                {"name": "done", "params": {"success": True}},
            ]
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["run", str(task_file), "--script", str(script_file)])
    assert result.exit_code == 1
    assert "success" in result.output.lower()
