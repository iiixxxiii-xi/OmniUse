"""Task 7.5 (report): markdown + CSV comparison tables for a suite result.

The report renders a :class:`SuiteResult` into a human-readable markdown summary
(the six aggregate metrics plus a per-task table) and a machine-readable CSV.
Both are pure functions of the suite, so they can be regenerated from a saved
results file without re-running the browser tasks.
"""

from minicua.eval.report import render_csv, render_markdown, write_report
from minicua.eval.runner import EvalResult, SuiteResult


def _suite() -> SuiteResult:
    results = [
        EvalResult(task_id="t1", score=1.0, success=True, steps=2, tool_calls=2, stop_reason="done"),
        EvalResult(task_id="t2", score=0.0, success=False, steps=1, tool_calls=1, stop_reason="done"),
    ]
    metrics = {
        "task_success": 0.5,
        "avg_tool_calls": 1.5,
        "token_cost": 0.0,
        "latency": 0.0,
        "recovery_rate": 0.0,
        "invalid_action_rate": 0.0,
    }
    return SuiteResult(results=results, metrics=metrics)


def test_render_markdown_has_summary_and_tasks():
    md = render_markdown(_suite(), title="CUA Eval Report")
    assert "# CUA Eval Report" in md
    assert "task_success" in md
    assert "0.5" in md
    assert "t1" in md and "t2" in md
    assert "PASS" in md and "FAIL" in md


def test_render_csv_header_and_rows():
    csv = render_csv(_suite())
    lines = csv.strip().splitlines()
    assert lines[0].startswith("task_id,success,score")
    assert any(line.startswith("t1,") for line in lines)
    assert any(line.startswith("t2,") for line in lines)
    assert any("true" in line for line in lines[1:])


def test_write_report_writes_files(tmp_path):
    md_path, csv_path = write_report(_suite(), tmp_path)
    assert md_path.name == "report.md" and md_path.is_file()
    assert csv_path.name == "report.csv" and csv_path.is_file()
    assert "task_success" in md_path.read_text(encoding="utf-8")
    assert "task_id,success" in csv_path.read_text(encoding="utf-8")


def test_render_markdown_empty_suite():
    md = render_markdown(SuiteResult())
    assert "0 tasks" in md
