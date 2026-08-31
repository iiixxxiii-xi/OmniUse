"""Eval report: markdown + CSV comparison tables for a :class:`SuiteResult`.

Both renderers are pure functions of the suite, so a saved results file can be
regenerated into a report without re-running the browser. The markdown is a
human summary (six aggregate metrics + a per-task table); the CSV is the
machine-readable comparison table (one row per task, full fidelity columns).
"""

import csv
import io
from pathlib import Path

from minicua.eval.runner import EvalResult, SuiteResult

#: Aggregate metric display order.
_METRIC_ORDER = (
    "task_success",
    "avg_tool_calls",
    "token_cost",
    "latency",
    "recovery_rate",
    "invalid_action_rate",
)

#: Per-task markdown columns (header, attribute name).
_MD_COLUMNS = (
    ("Task", "task_id"),
    ("Success", "success"),
    ("Score", "score"),
    ("Steps", "steps"),
    ("Tool Calls", "tool_calls"),
    ("Tokens", "tokens"),
    ("Cost (USD)", "cost_usd"),
    ("Latency (s)", "latency_seconds"),
    ("Recoveries", "recoveries"),
    ("Stop Reason", "stop_reason"),
)

#: Per-task CSV columns (full fidelity, one row per task).
_CSV_COLUMNS = (
    "task_id",
    "success",
    "score",
    "steps",
    "tool_calls",
    "tokens",
    "cost_usd",
    "latency_seconds",
    "recoveries",
    "page_changes",
    "stop_reason",
    "submission",
    "error",
)


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _md_cell(value: object) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    return _fmt(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(suite: SuiteResult, title: str = "CUA Eval Report") -> str:
    """Render a human-readable markdown report for ``suite``."""
    lines: list[str] = [f"# {title}", ""]
    lines.append(
        f"{suite.n_total} tasks, {suite.n_passed} passed "
        f"({suite.success_rate * 100:.1f}% success)"
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| ------ | ----- |")
    for name in _METRIC_ORDER:
        lines.append(f"| {name} | {_fmt(suite.metrics.get(name))} |")
    lines.append("")

    lines.append("## Tasks")
    lines.append("")
    lines.append("| " + " | ".join(header for header, _ in _MD_COLUMNS) + " |")
    lines.append("| " + " | ".join("---" for _ in _MD_COLUMNS) + " |")
    for result in suite.results:
        cells = [_md_cell(getattr(result, attr)) for _, attr in _MD_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_csv(suite: SuiteResult) -> str:
    """Render the per-task comparison table as CSV."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(list(_CSV_COLUMNS))
    for result in suite.results:
        writer.writerow([_fmt(getattr(result, attr)) for attr in _CSV_COLUMNS])
    return buffer.getvalue()


def write_report(
    suite: SuiteResult,
    output_dir: str | Path,
    title: str = "CUA Eval Report",
) -> tuple[Path, Path]:
    """Write ``report.md`` and ``report.csv`` into ``output_dir`` (created as needed).

    Returns the two written paths ``(markdown, csv)``.
    """
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    md_path = target / "report.md"
    csv_path = target / "report.csv"
    md_path.write_text(render_markdown(suite, title=title), encoding="utf-8")
    csv_path.write_text(render_csv(suite), encoding="utf-8")
    return md_path, csv_path
