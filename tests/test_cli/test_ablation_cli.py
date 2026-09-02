"""The ``ablation`` CLI command."""

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


def test_ablation_help():
    result = runner.invoke(app, ["ablation", "--help"])
    assert result.exit_code == 0
    assert "recovery" in result.output.lower()


def test_ablation_writes_comparison_files(tmp_path):
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
    result = runner.invoke(
        app, ["ablation", str(tasks_dir), "--output", str(out_dir), "--script", str(script_file)]
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "ablation.md").is_file()
    assert (out_dir / "ablation.json").is_file()
    # The JSON comparison carries both modes plus the delta metrics.
    data = json.loads((out_dir / "ablation.json").read_text(encoding="utf-8"))
    assert "baseline" in data and "full" in data
    assert data["baseline"]["results"][0]["task_id"] == "t1"
    assert data["full"]["results"][0]["task_id"] == "t1"
