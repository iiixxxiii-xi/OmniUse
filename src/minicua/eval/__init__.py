"""Eval layer: declarative evaluator, six-run metrics, task runner, and reports."""

from minicua.eval.ablation import AblationResult, run_ablation
from minicua.eval.errors import (
    EvalError,
    EvaluatorError,
    GetterError,
    MetricError,
    TaskDefinitionError,
)
from minicua.eval.evaluator import EvaluatorSpec, evaluate
from minicua.eval.getters import GETTERS, get_getter
from minicua.eval.metrics import METRICS, get_metric
from minicua.eval.metrics_aggregate import SIX_METRICS, aggregate
from minicua.eval.report import render_csv, render_markdown, write_report
from minicua.eval.runner import EvalResult, SuiteResult, run_suite, run_task
from minicua.eval.task import TaskDef, load_task_file, load_tasks

__all__ = [
    "AblationResult",
    "EvalError",
    "EvalResult",
    "EvaluatorError",
    "EvaluatorSpec",
    "GETTERS",
    "METRICS",
    "GetterError",
    "MetricError",
    "SIX_METRICS",
    "SuiteResult",
    "TaskDef",
    "TaskDefinitionError",
    "aggregate",
    "evaluate",
    "get_getter",
    "get_metric",
    "load_task_file",
    "load_tasks",
    "render_csv",
    "render_markdown",
    "run_ablation",
    "run_suite",
    "run_task",
    "write_report",
]
