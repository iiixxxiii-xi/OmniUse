"""The self-contained browser task set (``tasks/*.json``).

These tests lock the task set in: it must load cleanly, have a sane size
(15-25 tasks), carry unique ids, and at least one task must be solvable
end-to-end by a scripted :class:`FakeModel` — proving the declarative
fixtures + evaluators actually describe a reachable goal.
"""

from pathlib import Path

import pytest

from minicua.controller.llm import FakeModel
from minicua.eval.runner import run_task
from minicua.eval.task import load_tasks

TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks"


def _load() -> list:
    return load_tasks(TASKS_DIR)


def test_task_set_size():
    tasks = _load()
    assert 15 <= len(tasks) <= 25


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
