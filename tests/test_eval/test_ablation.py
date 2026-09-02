"""Baseline-vs-full recovery ablation: ``run_ablation`` + ``AblationResult``.

The ablation runs the *same* task set twice — once as a bare ReAct loop
(``recovery=False``) and once with the full recovery ladder (``recovery=True``) —
and reports the success-rate / invalid-action / recovery-success deltas.
"""

import pytest

from minicua.controller.llm import FakeModel
from minicua.eval.ablation import AblationResult, run_ablation
from minicua.eval.runner import EvalResult, SuiteResult
from minicua.eval.task import TaskDef


def _nav_click_task() -> TaskDef:
    return TaskDef(
        id="nav_click",
        instruction="navigate then click",
        html="<button id=start>start</button>",
        evaluator={
            "func": "contains",
            "result": {"getter": "page_text"},
            "expected": {"expected": "ok"},
        },
    )


# One multi-action step (navigate + a click grounded on the pre-nav page) then done.
# Full mode aborts the click via the page-change guard; baseline executes it stale.
_NAV_CLICK_SCRIPT = [
    [
        {"name": "navigate", "params": {"url": "data:text/html,<button id=b>ok</button>"}},
        {"name": "click", "params": {"index": 1}},
    ],
    {"name": "done", "params": {"success": True}},
]


@pytest.mark.asyncio
async def test_run_ablation_produces_baseline_and_full():
    tasks = [_nav_click_task()]
    result = await run_ablation(tasks, lambda: FakeModel(responses=list(_NAV_CLICK_SCRIPT)))

    assert isinstance(result, AblationResult)
    assert result.baseline.n_total == 1
    assert result.full.n_total == 1

    # Both modes pass (the navigate succeeded and the evaluator sees "ok").
    assert result.baseline.success_rate == 1.0
    assert result.full.success_rate == 1.0
    assert result.success_rate_delta == 0.0

    # Baseline executed the stale click (an invalid action); full aborted it.
    assert result.baseline.metrics["invalid_action_rate"] == pytest.approx(1 / 3)
    assert result.full.metrics["invalid_action_rate"] == 0.0
    assert result.invalid_action_delta == pytest.approx(1 / 3)

    # The page-change guard fired only in full mode.
    assert result.baseline.results[0].page_changes == 0
    assert result.full.results[0].page_changes == 1

    # No stale/crash recovery events in this task, so recovery success is 0.
    assert result.recovery_success_rate == 0.0


def test_ablation_recovery_success_rate_from_attempts():
    full = SuiteResult(
        results=[
            EvalResult(
                task_id="a", score=1.0, success=True, stop_reason="done",
                recoveries=2, recovery_attempts=4,
            ),
            EvalResult(
                task_id="b", score=0.0, success=False, stop_reason="done",
                recoveries=0, recovery_attempts=0,
            ),
        ],
        metrics={},
    )
    ablation = AblationResult(baseline=SuiteResult(), full=full)
    assert ablation.recovery_success_rate == 0.5


def test_ablation_success_rate_delta():
    baseline = SuiteResult(
        results=[EvalResult(task_id="a", score=0.0, success=False, stop_reason="done")],
        metrics={},
    )
    full = SuiteResult(
        results=[EvalResult(task_id="a", score=1.0, success=True, stop_reason="done")],
        metrics={},
    )
    ablation = AblationResult(baseline=baseline, full=full)
    assert ablation.success_rate_delta == 1.0


def test_ablation_invalid_action_delta():
    baseline = SuiteResult(results=[], metrics={"invalid_action_rate": 0.4})
    full = SuiteResult(results=[], metrics={"invalid_action_rate": 0.1})
    ablation = AblationResult(baseline=baseline, full=full)
    assert ablation.invalid_action_delta == pytest.approx(0.3)
