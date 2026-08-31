"""Task 7.5: eval runner — run one task (setup → agent → evaluate) and a suite.

The runner wires the pieces together: it starts a browser (or uses a caller
session), sets the page up from the task's ``html`` fixture or ``initial_url``,
drives the :class:`Agent` with the given model, scores the result with the
declarative evaluator, and records a single-run :class:`EvalResult`. A
:class:`SuiteResult` aggregates the six metrics across a whole task set.
"""

import json

import pytest

from minicua.controller.agent import AgentResult, StopReason
from minicua.controller.llm import FakeModel
from minicua.eval.errors import TaskDefinitionError
from minicua.eval.runner import SuiteResult, event_log_from_result, run_suite, run_task
from minicua.eval.task import TaskDef, load_tasks
from minicua.state.events import ActionEvent

CLICK_THEN_DONE = [
    {"name": "click", "params": {"index": 1}},
    {"name": "done", "params": {"success": True}},
]


def _click_task() -> TaskDef:
    return TaskDef(
        id="t1",
        instruction="click go",
        html=(
            "<button id=go onclick=\"document.getElementById('out').textContent='clicked'\">go</button>"
            "<div id=out></div>"
        ),
        evaluator={
            "func": "exact_match",
            "conj": "and",
            "result": {"getter": "element_text", "selector": "#out"},
            "expected": {"expected": "clicked"},
        },
    )


@pytest.mark.asyncio
async def test_runner_one_task_succeeds(session):
    result = await run_task(_click_task(), FakeModel(responses=list(CLICK_THEN_DONE)), session=session)
    assert result.task_id == "t1"
    assert result.success is True
    assert result.score == 1.0
    assert result.stop_reason == StopReason.DONE.value
    assert result.tool_calls == 2  # click + done


@pytest.mark.asyncio
async def test_runner_wrong_action_fails(session):
    # The model clicks index 999 (no such element), so the evaluator scores 0.
    task = _click_task()
    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 999}},
            {"name": "done", "params": {"success": True}},
        ]
    )
    result = await run_task(task, model, session=session)
    assert result.success is False
    assert result.score == 0.0
    assert result.stop_reason == StopReason.DONE.value  # agent finished; the *evaluator* failed


@pytest.mark.asyncio
async def test_runner_uses_initial_url(session):
    task = TaskDef(
        id="url_task",
        instruction="verify text",
        initial_url="data:text/html,<div>hello world</div>",
        evaluator={"func": "contains", "result": {"getter": "page_text"}, "expected": {"expected": "hello"}},
    )
    result = await run_task(task, FakeModel(responses=[{"name": "done", "params": {"success": True}}]), session=session)
    assert result.success is True
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_runner_model_error_is_structured_not_crash(session):
    # A model that runs out of script fails the run but the runner still returns a result.
    task = _click_task()
    result = await run_task(task, FakeModel(responses=[]), session=session)
    assert result.success is False
    assert result.score == 0.0
    assert result.stop_reason == StopReason.MODEL_ERROR.value
    assert result.error is not None


@pytest.mark.asyncio
async def test_run_suite_aggregates(session):
    good = _click_task()
    bad = _click_task().model_copy(update={"id": "t2", "evaluator": {"func": "exact_match", "result": {"getter": "element_text", "selector": "#out"}, "expected": {"expected": "nope"}}})
    suite = await run_suite(
        [good, bad],
        FakeModel(responses=list(CLICK_THEN_DONE)),
    )
    assert isinstance(suite, SuiteResult)
    assert len(suite.results) == 2
    assert suite.n_passed == 1
    assert suite.metrics["task_success"] == 0.5


# --------------------------------------------------------------------------- #
# event log synthesis
# --------------------------------------------------------------------------- #


def test_event_log_from_result_counts_actions():
    result = AgentResult(
        done=True,
        success=True,
        stop_reason=StopReason.DONE,
        steps=2,
        history=[
            {"step": 1, "actions": [{"name": "click", "params": {"index": 1}}], "results": [{"success": True}]},
            {"step": 2, "actions": [{"name": "done", "params": {"success": True}}], "results": [{"success": True}]},
        ],
    )
    log = event_log_from_result(result, latency_seconds=3.0)
    actions = [e for e in log.events if isinstance(e, ActionEvent)]
    assert len(actions) == 2
    # latency is encoded as the ts span (first event 0.0, last event 3.0).
    timestamps = [e.ts for e in log.events]
    assert max(timestamps) - min(timestamps) == 3.0


# --------------------------------------------------------------------------- #
# task loading
# --------------------------------------------------------------------------- #


def test_load_tasks_from_dir(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps(
            {
                "id": "a",
                "instruction": "do a",
                "html": "<div>x</div>",
                "evaluator": {"func": "exact_match", "result": {"getter": "page_text"}, "expected": {"expected": "x"}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "b.json").write_text(
        json.dumps(
            {
                "id": "b",
                "instruction": "do b",
                "evaluator": {"func": "contains", "result": {"getter": "page_text"}, "expected": {"expected": "y"}},
            }
        ),
        encoding="utf-8",
    )
    tasks = load_tasks(tmp_path)
    assert [t.id for t in tasks] == ["a", "b"]
    assert all(isinstance(t, TaskDef) for t in tasks)


def test_load_tasks_tolerates_corrupt_file(tmp_path):
    (tmp_path / "good.json").write_text(
        json.dumps(
            {
                "id": "good",
                "instruction": "x",
                "evaluator": {"func": "exact_match", "result": {"getter": "page_url"}, "expected": {"expected": "x"}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "corrupt.json").write_text("{ not valid json", encoding="utf-8")
    tasks = load_tasks(tmp_path)  # default: skip corrupt, don't raise
    assert [t.id for t in tasks] == ["good"]


def test_load_tasks_strict_raises_on_corrupt(tmp_path):
    (tmp_path / "corrupt.json").write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(TaskDefinitionError):
        load_tasks(tmp_path / "corrupt.json", strict=True)


def test_load_tasks_missing_path_raises(tmp_path):
    with pytest.raises(TaskDefinitionError):
        load_tasks(tmp_path / "does_not_exist.json")


def test_load_tasks_missing_instruction_raises(tmp_path):
    (tmp_path / "bad.json").write_text(json.dumps({"id": "x"}), encoding="utf-8")
    with pytest.raises(TaskDefinitionError):
        load_tasks(tmp_path / "bad.json")
