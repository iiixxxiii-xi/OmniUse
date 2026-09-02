"""Recovery stress tasks: each recovery rung fires deterministically.

The problem: with a capable model the recovery ladder never trips (0 triggers in
ablation), so its value went unverified. These tasks manufacture a *transient
fault* in the page itself — a re-render, a navigation, a crash, a dead button, a
disabled submit — so the recovery layer is exercised by the task, not by hoping
the model makes a mistake.

Each test drives a scripted :class:`FakeModel` through the fault on a real
Playwright page and asserts *both* halves of the guarantee:

* the recovery fired (``recovery_attempts`` / ``recoveries`` / ``page_changes``,
  or a loop/replan nudge reached the model);
* the task still lands on a passing evaluator score.

Layers covered:

* **stale** (``recovery_stale_*``) — the page removes/reorders elements between
  perception and action; the stale index is relocalized and re-executed.
* **page-change** (``recovery_page_change_*``) — a multi-action step whose first
  action navigates/re-renders aborts the remaining queue.
* **crash** (``recovery_crash_*``) — the session is rebuilt from checkpoint and
  storage_state (a cookie) survives.
* **loop** (``recovery_loop_*``) — a dead button makes the page stagnate; a soft
  nudge is injected.
* **replan-on-stall** (``recovery_replan_*``) — two consecutive failures push a
  ``REPLAN SUGGESTED`` nudge before the model changes strategy.
"""

from pathlib import Path

import pytest

from minicua.controller.agent import Agent
from minicua.controller.llm import FakeModel
from minicua.eval.evaluator import evaluate
from minicua.eval.runner import run_task
from minicua.eval.task import load_tasks

TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks"
_FIXTURE_URL = "http://minicua.local/"

#: The recovery stress task ids, grouped by the recovery rung they exercise.
STRESS_TASKS = {
    "stale": [
        "recovery_stale_filter_list",
        "recovery_stale_remove_item",
        "recovery_stale_auto_collapse",
    ],
    "page_change": [
        "recovery_page_change_navigate",
        "recovery_page_change_rerender",
    ],
    "crash": ["recovery_crash_cookie_resume"],
    "loop": ["recovery_loop_stagnation"],
    "replan": ["recovery_replan_disabled_submit"],
}


def _task(task_id):
    return next(t for t in load_tasks(TASKS_DIR) if t.id == task_id)


def _user_texts(model):
    """All plain-text user messages the model saw, across every call."""
    texts = []
    for messages, _ in model.calls:
        for m in messages:
            if m.role != "user":
                continue
            if isinstance(m.content, str):
                texts.append(m.content)
            else:
                texts.append("".join(b.text for b in m.content if getattr(b, "type", None) == "text"))
    return texts


# --------------------------------------------------------------------------- #
# task-set invariants
# --------------------------------------------------------------------------- #


def test_recovery_stress_tasks_present_and_declarative():
    ids = [i for group in STRESS_TASKS.values() for i in group]
    tasks = {t.id: t for t in load_tasks(TASKS_DIR)}
    assert len(ids) == 8
    assert len(set(ids)) == len(ids)
    for tid in ids:
        task = tasks[tid]
        assert task.vision_required is False, f"{tid}: must be DOM-graded"
        assert task.html is not None, f"{tid}: must be self-contained"
        assert task.evaluator is not None, f"{tid}: must carry a declarative evaluator"


# --------------------------------------------------------------------------- #
# stale-element relocalization
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stale_filter_list_relocalizes_via_ax_name(session):
    # Filter removes Alpha/Beta, so "Save" (old index 3) is relocalized to index 1
    # by accessible name; recovery re-executes the click successfully.
    task = _task("recovery_stale_filter_list")
    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 1}},  # Filter -> list collapses
            {"name": "click", "params": {"index": 3}},  # stale Save index
            {"name": "done", "params": {"success": True}},
        ]
    )
    result = await run_task(task, model, session=session)
    assert result.success is True
    assert result.score == 1.0
    assert result.recovery_attempts == 1
    assert result.recoveries == 1


@pytest.mark.asyncio
async def test_stale_remove_item_relocalizes_via_stable_hash(session):
    # Removing "Item 1" shifts "Confirm" from index 3 to 2; its id-based xpath (and
    # hence stable_hash) is unchanged, so recovery relocalizes by the strongest signal.
    task = _task("recovery_stale_remove_item")
    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 1}},  # Remove first
            {"name": "click", "params": {"index": 3}},  # stale Confirm index
            {"name": "done", "params": {"success": True}},
        ]
    )
    result = await run_task(task, model, session=session)
    assert result.success is True
    assert result.score == 1.0
    assert result.recovery_attempts == 1
    assert result.recoveries == 1


@pytest.mark.asyncio
async def test_stale_auto_collapse_relocalizes_after_timeout(session):
    # A setTimeout collapses the list on its own; a `wait` lets it fire, then the
    # model's cached index is stale and recovery relocalizes via the previous state.
    task = _task("recovery_stale_auto_collapse")
    model = FakeModel(
        responses=[
            {"name": "wait", "params": {"seconds": 2}},  # let setTimeout fire
            {"name": "click", "params": {"index": 3}},  # stale Save index
            {"name": "done", "params": {"success": True}},
        ]
    )
    result = await run_task(task, model, session=session)
    assert result.success is True
    assert result.score == 1.0
    assert result.recovery_attempts == 1
    assert result.recoveries == 1


# --------------------------------------------------------------------------- #
# page-change guard (multi-action abort)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_page_change_navigate_aborts_stale_click(session):
    # One step emits navigate + click; the navigate moves the page, so the guard
    # aborts the click grounded on the pre-navigation page. The next step clicks
    # the fresh submit button and passes.
    task = _task("recovery_page_change_navigate")
    model = FakeModel(
        responses=[
            [
                {"name": "navigate", "params": {"url": _FIXTURE_URL + "#details"}},
                {"name": "click", "params": {"index": 1}},
            ],
            {"name": "click", "params": {"index": 3}},  # submit on the details view
            {"name": "done", "params": {"success": True}},
        ]
    )
    result = await run_task(task, model, session=session)
    assert result.success is True
    assert result.score == 1.0
    assert result.page_changes == 1


@pytest.mark.asyncio
async def test_page_change_rerender_aborts_stale_click(session):
    # Clicking Load re-renders the whole content region (element count + text hash
    # change) with no navigation; the guard aborts the same-step Confirm click.
    task = _task("recovery_page_change_rerender")
    model = FakeModel(
        responses=[
            [
                {"name": "click", "params": {"index": 1}},  # Load
                {"name": "click", "params": {"index": 2}},  # Confirm (stale, aborted)
            ],
            {"name": "click", "params": {"index": 2}},  # Confirm (fresh)
            {"name": "done", "params": {"success": True}},
        ]
    )
    result = await run_task(task, model, session=session)
    assert result.success is True
    assert result.score == 1.0
    assert result.page_changes == 1


# --------------------------------------------------------------------------- #
# crash recovery
# --------------------------------------------------------------------------- #


class _CrashInjectModel(FakeModel):
    """Sets the crash watchdog flag after the Nth model call (mid-run crash)."""

    def __init__(self, responses, crash_after_call=1):
        super().__init__(responses)
        self.crash_after_call = crash_after_call
        self.agent = None

    async def generate(self, messages, tools):
        result = await super().generate(messages, tools)
        if len(self.calls) == self.crash_after_call and self.agent is not None:
            self.agent._watchdog.crashed = True
        return result


@pytest.mark.asyncio
async def test_crash_recovers_and_cookie_survives(session, tmp_path):
    # The login click sets a cookie, which is checkpointed to storage_state. A
    # mid-run crash rebuilds the session; recovery restores the cookie and the
    # task, and the evaluator passes on the restored state.
    task = _task("recovery_crash_cookie_resume")

    async def handler(route):
        await route.fulfill(status=200, content_type="text/html", body=task.html)

    await session.context.route(_FIXTURE_URL + "**", handler)
    await session.page.goto(_FIXTURE_URL)

    model = _CrashInjectModel(
        responses=[
            {"name": "click", "params": {"index": 1}},  # Login -> sets cookie
            {"name": "done", "params": {"success": True}},
        ],
        crash_after_call=1,
    )
    agent = Agent(
        session=session,
        model=model,
        task=task.instruction,
        checkpoint_dir=tmp_path,
        max_steps=10,
        max_failures=5,
    )
    model.agent = agent
    result = await agent.run(task.instruction)

    assert result.done is True
    assert result.recovery_attempts == 1
    assert result.recoveries == 1
    assert agent.task == task.instruction  # checkpoint restored the task

    score = await evaluate(session, task.evaluator)
    assert score == 1.0  # cookie survived via storage_state


# --------------------------------------------------------------------------- #
# loop detection (soft nudge)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_loop_stagnation_injects_nudge(session):
    # The "Refresh" decoy does nothing, so five clicks leave the page unchanged.
    # The loop detector injects a repetition/stagnation nudge, the model then
    # clicks the real Submit button, and the task passes.
    task = _task("recovery_loop_stagnation")
    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 1}},  # Refresh (no-op)
            {"name": "click", "params": {"index": 1}},
            {"name": "click", "params": {"index": 1}},
            {"name": "click", "params": {"index": 1}},
            {"name": "click", "params": {"index": 1}},
            {"name": "click", "params": {"index": 2}},  # Submit
            {"name": "done", "params": {"success": True}},
        ]
    )
    result = await run_task(task, model, session=session)
    assert result.success is True
    assert result.score == 1.0

    nudges = [
        t for t in _user_texts(model)
        if "repeated" in t or "unchanged" in t
    ]
    assert nudges


# --------------------------------------------------------------------------- #
# replan-on-stall
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_replan_disabled_submit_injects_replan_nudge(session):
    # Two failed clicks on the disabled Submit (non-stale failures) push a
    # "REPLAN SUGGESTED" nudge; the model then checks the box and succeeds.
    task = _task("recovery_replan_disabled_submit")
    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 2}},  # disabled Submit -> fail
            {"name": "click", "params": {"index": 2}},  # disabled Submit -> fail -> replan
            {"name": "click", "params": {"index": 1}},  # checkbox -> enable
            {"name": "click", "params": {"index": 2}},  # Submit -> success
            {"name": "done", "params": {"success": True}},
        ]
    )
    result = await run_task(task, model, session=session)
    assert result.success is True
    assert result.score == 1.0

    nudges = [t for t in _user_texts(model) if "REPLAN SUGGESTED" in t]
    assert nudges
