"""Task 8.2: the ``eval`` and ``report`` CLI commands."""

import json

from typer.testing import CliRunner

from minicua.cli.main import app
from minicua.eval.runner import EvalResult, SuiteResult

runner = CliRunner()

_CLICK_TASK = {
    "id": "t1",
    "instruction": "click the button",
    "html": "<button id=btn onclick=\"document.getElementById('out').textContent='clicked'\">go</button><div id=out></div>",
    "evaluator": {"func": "exact_match", "result": {"getter": "element_text", "selector": "#out"}, "expected": {"expected": "clicked"}},
}


def test_eval_help():
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "tasks" in result.output.lower()


def test_report_help():
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code == 0
    assert "results" in result.output.lower()


def test_eval_writes_report_files(tmp_path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "t1.json").write_text(json.dumps(_CLICK_TASK), encoding="utf-8")
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
    out_dir = tmp_path / "out"
    result = runner.invoke(app, ["eval", str(tasks_dir), "--output", str(out_dir), "--script", str(script_file)])
    assert result.exit_code == 0, result.output
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "report.csv").is_file()
    assert (out_dir / "results.json").is_file()


def test_eval_missing_tasks_dir_errors(tmp_path):
    result = runner.invoke(app, ["eval", str(tmp_path / "does_not_exist")])
    assert result.exit_code != 0
    assert "error" in result.output.lower()


def test_report_rerenders_from_results(tmp_path):
    suite = SuiteResult(
        results=[EvalResult(task_id="t1", score=1.0, success=True, stop_reason="done")],
        metrics={"task_success": 1.0, "avg_tool_calls": 1.0, "token_cost": 0.0, "latency": 0.0, "recovery_rate": 0.0, "invalid_action_rate": 0.0},
    )
    results_file = tmp_path / "results.json"
    results_file.write_text(suite.model_dump_json(), encoding="utf-8")
    out_dir = tmp_path / "out"
    result = runner.invoke(app, ["report", str(results_file), "--output", str(out_dir)])
    assert result.exit_code == 0, result.output
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "report.csv").is_file()
