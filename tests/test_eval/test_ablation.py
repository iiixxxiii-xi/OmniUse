"""Baseline-vs-full recovery ablation: ``run_ablation`` + ``AblationResult``.

The ablation runs the *same* task set twice — once as a bare ReAct loop
(``recovery=False``) and once with the full recovery ladder (``recovery=True``) —
and reports the success-rate / invalid-action / recovery-success deltas.
"""

import pytest

from minicua.controller.llm import FakeModel, ModelOutput, ToolCall
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


def _requery_task() -> TaskDef:
    return TaskDef(
        id="requery",
        instruction="navigate to the target page",
        html="<button id=start>start</button>",
        evaluator={
            "func": "contains",
            "result": {"getter": "page_text"},
            "expected": {"expected": "ok"},
        },
    )


# The model's first response is malformed (no tool calls). Full mode requeries and
# then navigates to the target page; baseline fails outright before reaching it.
_REQUERY_SCRIPT = [
    ModelOutput(),  # malformed -> requery (full) / fail (baseline)
    {"name": "navigate", "params": {"url": "data:text/html,<div>ok</div>"}},
    {"name": "done", "params": {"success": True}},
]


@pytest.mark.asyncio
async def test_run_ablation_requery_distinguishes_baseline_from_full():
    tasks = [_requery_task()]
    result = await run_ablation(tasks, lambda: FakeModel(responses=list(_REQUERY_SCRIPT)))

    # Baseline (bare ReAct) fails on the malformed first response and never reaches
    # the page that would satisfy the evaluator; full mode requeries and succeeds.
    assert result.baseline.success_rate == 0.0
    assert result.full.success_rate == 1.0
    assert result.success_rate_delta == 1.0
    assert result.baseline.results[0].stop_reason == "invalid_response"
    assert result.full.results[0].stop_reason == "done"


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


# --------------------------------------------------------------------------- #
# re-observe + re-plan recovery rung
# --------------------------------------------------------------------------- #


def _text_of(message) -> str:
    """Plain-text rendering of one message (handles str and content-block shapes)."""
    if isinstance(message.content, str):
        return message.content
    return "".join(b.text for b in message.content if getattr(b, "type", None) == "text")


def _last_user_text(messages) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return _text_of(m)
    return ""


class StaleRecoveryModel:
    """A model that hallucinates a stale index on normal turns but corrects itself
    when the recovery layer feeds back the re-plan hint (fresh DOM + error).

    On a *normal* turn (the last user message is a fresh state observation) it
    emits ``click <stale_index>`` up to ``max_stale`` times, then ``done``. When
    the last user message is the re-plan hint, it emits the correct index instead.
    This models a model that keeps using a cached index until it is re-observed
    and re-planned — exactly what bare ReAct fails on and full recovery rescues.
    """

    supports_vision = False

    def __init__(self, *, stale_index: int, correct_index: int, max_stale: int) -> None:
        self.stale_index = stale_index
        self.correct_index = correct_index
        self.max_stale = max_stale
        self._normal = 0
        self.calls: list = []

    async def generate(self, messages, tools):
        self.calls.append(tuple(messages))
        last = _last_user_text(messages)
        if "Re-plan using the fresh page elements" in last:
            return ModelOutput(tool_calls=[ToolCall(name="click", arguments={"index": self.correct_index})])
        self._normal += 1
        if self._normal <= self.max_stale:
            return ModelOutput(tool_calls=[ToolCall(name="click", arguments={"index": self.stale_index})])
        return ModelOutput(tool_calls=[ToolCall(name="done", arguments={"success": True})])


def _replan_task() -> TaskDef:
    return TaskDef(
        id="replan_stale_index",
        instruction="click the submit button",
        html=(
            "<button id=\"submit\" onclick=\"document.getElementById('out').textContent='ok'\">go</button>"
            "<div id=\"out\"></div>"
        ),
        evaluator={
            "func": "exact_match",
            "result": {"getter": "element_text", "selector": "#out"},
            "expected": {"expected": "ok"},
        },
    )


@pytest.mark.asyncio
async def test_run_ablation_replan_recovery_distinguishes_baseline_from_full():
    tasks = [_replan_task()]
    result = await run_ablation(
        tasks,
        lambda: StaleRecoveryModel(stale_index=999, correct_index=1, max_stale=3),
    )

    # Bare ReAct keeps hallucinating index 999 and exhausts max_failures; full mode
    # re-observes + re-plans each failure, clicks the correct element, and finishes.
    assert result.baseline.success_rate == 0.0
    assert result.full.success_rate == 1.0
    assert result.success_rate_delta == 1.0
    assert result.baseline.results[0].stop_reason == "max_failures"
    assert result.full.results[0].stop_reason == "done"

    # Recovery succeeded on every attempt (re-observe + re-plan rescues each one).
    assert result.recovery_success_rate == 1.0
    assert result.full.results[0].recoveries == 3
    assert result.full.results[0].recovery_attempts == 3
