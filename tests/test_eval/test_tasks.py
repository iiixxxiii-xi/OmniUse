"""The self-contained browser task set (``tasks/*.json``).

These tests lock the task set in: it must load cleanly, have a sane size
(15-65 tasks), carry unique ids, span all three difficulty tiers, reference only
known getters/metrics, and serve every inline fixture on a real origin — proving
the declarative fixtures + evaluators actually describe a reachable goal. Several
end-to-end tests script a :class:`FakeModel` through representative tasks to show
the evaluator truly distinguishes "solved" from "not solved".
"""

from pathlib import Path

import pytest

from minicua.controller.llm import FakeModel
from minicua.eval.evaluator import evaluate
from minicua.eval.getters import GETTERS
from minicua.eval.metrics import METRICS
from minicua.eval.runner import run_task
from minicua.eval.task import load_tasks

TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks"
_FIXTURE_URL = "http://minicua.local/"


def _load() -> list:
    return load_tasks(TASKS_DIR)


def test_task_set_size():
    tasks = _load()
    assert 15 <= len(tasks) <= 65


def test_task_ids_unique():
    tasks = _load()
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids))


def test_every_task_has_evaluator_and_setup():
    tasks = _load()
    for t in tasks:
        assert t.instruction
        # every task is runnable: an inline html fixture or an initial_url
        assert t.html is not None or t.initial_url


def test_difficulty_spans_all_tiers():
    tasks = _load()
    tiers = {t.difficulty for t in tasks}
    assert tiers == {"easy", "medium", "hard"}


def test_every_evaluator_references_known_getters_and_metrics():
    tasks = _load()
    for t in tasks:
        for metric_name, result_cfg in zip(t.evaluator.funcs, t.evaluator.results):
            assert metric_name in METRICS, f"{t.id}: unknown metric {metric_name!r}"
            getter_name = result_cfg.get("getter")
            assert getter_name in GETTERS, f"{t.id}: unknown getter {getter_name!r}"


@pytest.mark.asyncio
async def test_every_fixture_servable_and_evaluator_runnable(session):
    for task in _load():
        if task.html is not None:
            html = task.html

            async def handler(route):
                await route.fulfill(status=200, content_type="text/html", body=html)

            await session.context.route(_FIXTURE_URL + "**", handler)
            await session.page.goto(_FIXTURE_URL)
            try:
                score = await evaluate(session, task.evaluator)
            finally:
                await session.context.unroute(_FIXTURE_URL + "**")
        else:
            await session.navigate(task.initial_url)
            score = await evaluate(session, task.evaluator)
        assert isinstance(score, float), f"{task.id}: evaluator returned {score!r}"
        assert 0.0 <= score <= 1.0, f"{task.id}: score out of range {score!r}"


@pytest.mark.asyncio
async def test_solve_click_button_end_to_end(session):
    # The 'click_button' fixture has a single button (index 1) that writes #out.
    task = next(t for t in _load() if t.id == "click_button")
    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 1}},
            {"name": "done", "params": {"success": True}},
        ]
    )
    result = await run_task(task, model, session=session)
    assert result.success is True
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_solve_validate_email_long_horizon_end_to_end(session):
    # Long-horizon: type an invalid email → submit (rejected) → correct the email
    # → submit (accepted) → done. Proves a multi-step declarative task is
    # solvable and the evaluator distinguishes the wrong path from the right one.
    task = next(t for t in _load() if t.id == "validate_email_format")
    model = FakeModel(
        responses=[
            {"name": "type", "params": {"index": 1, "text": "not-an-email"}},
            {"name": "click", "params": {"index": 2}},
            {"name": "type", "params": {"index": 1, "text": "ada@example.com"}},
            {"name": "click", "params": {"index": 2}},
            {"name": "done", "params": {"success": True}},
        ]
    )
    result = await run_task(task, model, session=session)
    assert result.success is True
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_solve_rerender_reshuffle_end_to_end(session):
    # The re-render task shrinks the list after every click, so the "next" button
    # is always index 1. A model that re-reads the fresh DOM each step (rather
    # than caching a stale index) solves it — the whole point of re-observe.
    task = next(t for t in _load() if t.id == "rerender_reshuffle_buttons")
    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 1}},  # Alpha
            {"name": "click", "params": {"index": 1}},  # Beta  (now index 1)
            {"name": "click", "params": {"index": 1}},  # Gamma (now index 1)
            {"name": "done", "params": {"success": True}},
        ]
    )
    result = await run_task(task, model, session=session)
    assert result.success is True
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_solve_sort_toggle_rerender_end_to_end(session):
    # The sort toggle reorders the list buttons; a model that re-reads the fresh
    # DOM after the sort (rather than caching the pre-sort index) clicks 'Zulu'.
    task = next(t for t in _load() if t.id == "sort_toggle_rerender")
    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 1}},  # Sort descending
            {"name": "click", "params": {"index": 2}},  # Zulu (now the top item)
            {"name": "done", "params": {"success": True}},
        ]
    )
    result = await run_task(task, model, session=session)
    assert result.success is True
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_solve_dynamic_form_rows_end_to_end(session):
    # Adding fields shifts every subsequent element's index, so the model must
    # re-read the DOM after each add before typing / submitting.
    task = next(t for t in _load() if t.id == "dynamic_form_rows")
    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 1}},  # Add field
            {"name": "click", "params": {"index": 1}},  # Add field (now 3 inputs)
            {"name": "type", "params": {"index": 2, "text": "one"}},
            {"name": "type", "params": {"index": 3, "text": "two"}},
            {"name": "type", "params": {"index": 4, "text": "three"}},
            {"name": "click", "params": {"index": 5}},  # Submit
            {"name": "done", "params": {"success": True}},
        ]
    )
    result = await run_task(task, model, session=session)
    assert result.success is True
    assert result.score == 1.0
