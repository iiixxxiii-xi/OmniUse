"""Minimal (baseline ReAct) mode: the ``recovery`` master switch.

``Agent(recovery=False)`` is the bare perceive→think→act loop — no stale
relocalization, no page-change guard, no loop detection, no crash recovery. A
failed action fails outright; the controller never re-perceives and re-plans.
"""

import pytest

from minicua.controller.agent import Agent, StopReason
from minicua.controller.llm import FakeModel
from minicua.perception.extract import extract_state


@pytest.mark.asyncio
async def test_minimal_mode_stale_action_fails_without_relocalizing(session, monkeypatch):
    calls: list[str] = []

    async def fake_recover(action, old_state, page, previous_state=None):
        calls.append(action.name)
        return None

    monkeypatch.setattr("minicua.controller.agent.recover_stale", fake_recover)

    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 999}},  # stale, not in selector map
            {"name": "done", "params": {"success": True}},
        ]
    )
    agent = Agent(session=session, model=model, recovery=False, max_failures=5)
    result = await agent.run(task="x")

    assert result.done is True
    assert calls == []  # no relocalization was attempted
    assert result.recoveries == 0
    assert result.recovery_attempts == 0
    assert result.history[0].results[0].success is False  # failed outright


@pytest.mark.asyncio
async def test_full_mode_stale_action_relocalizes(session, monkeypatch):
    await session.page.set_content('<button id="a" aria-label="Save">save</button>')
    calls: list[str] = []

    async def fake_recover(action, old_state, page, previous_state=None):
        calls.append(action.name)
        relocalized = action.model_copy(
            update={"params": action.params.model_copy(update={"index": 1})}
        )
        return relocalized, await extract_state(page)

    monkeypatch.setattr("minicua.controller.agent.recover_stale", fake_recover)

    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 999}},  # stale
            {"name": "done", "params": {"success": True}},
        ]
    )
    agent = Agent(session=session, model=model, max_failures=5)
    result = await agent.run(task="x")

    assert result.done is True
    assert calls == ["click"]  # recovery was attempted
    assert result.recoveries == 1
    assert result.recovery_attempts == 1
    assert result.history[0].results[0].success is True  # relocalized + re-executed


@pytest.mark.asyncio
async def test_minimal_mode_disables_page_change_guard(session):
    model = FakeModel(
        responses=[
            [
                {"name": "navigate", "params": {"url": "data:text/html,<button id=b>ok</button>"}},
                {"name": "click", "params": {"index": 1}},  # grounded on the blank pre-nav page
            ],
        ]
    )
    agent = Agent(session=session, model=model, recovery=False, max_steps=10)
    step = await agent.step()

    assert step.is_done is False
    assert step.page_changed is False  # guard is off
    assert len(step.results) == 2  # both actions ran (click fails as stale)


@pytest.mark.asyncio
async def test_minimal_mode_disables_loop_nudge(session):
    await session.page.set_content("<button id=b>x</button>")
    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 1}},
            {"name": "click", "params": {"index": 1}},
            {"name": "click", "params": {"index": 1}},
            {"name": "done", "params": {"success": True}},
        ]
    )
    agent = Agent(session=session, model=model, recovery=False, max_steps=10, loop_threshold=2)
    result = await agent.run(task="x")

    assert result.done is True
    nudges = [
        m.content
        for call in model.calls
        for m in call[0]
        if m.role == "user" and "repeated" in m.content
    ]
    assert not nudges  # loop detection is off


@pytest.mark.asyncio
async def test_minimal_mode_disables_crash_recovery(session, tmp_path):
    await session.page.set_content("<button id=b>x</button>")
    model = FakeModel(responses=[{"name": "done", "params": {"success": True}}])
    agent = Agent(session=session, model=model, recovery=False, checkpoint_dir=tmp_path)

    # Simulate the watchdog flagging a browser crash before the first step.
    agent._watchdog.crashed = True

    result = await agent.run(task="book a flight")
    assert result.done is False
    assert result.stop_reason == StopReason.ERROR  # crash was not recovered
