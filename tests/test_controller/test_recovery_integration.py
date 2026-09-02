"""Stage 5 integration: the recovery layer wired into the agent loop.

These tests exercise the four recovery strategies end-to-end:

* **stale** — ``recover_stale`` re-perceives and relocalizes a moved element; the
  agent invokes it on a retryable ``STALE_ELEMENT`` / ``ELEMENT_NOT_FOUND``.
* **page change** — a multi-action step whose first action navigates aborts the
  remaining queue instead of clicking on stale coordinates.
* **loop** — a soft nudge message reaches the model when it repeats itself.
* **crash** — a crashed session is rebuilt and the task state restored.
"""

import pytest

from minicua.action.executor import execute
from minicua.action.models import Action, ClickParams
from minicua.controller.agent import Agent
from minicua.controller.llm import FakeModel
from minicua.perception.extract import extract_state
from minicua.recovery.stale import recover_stale


# --------------------------------------------------------------------------- #
# stale element recovery
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_recover_stale_relocalizes_and_reexecutes(session):
    # The page re-renders: the "Save" button (by accessible name) moves from
    # index 1 to index 2, with a different id/xpath, so only ax_name can match.
    await session.page.set_content('<button id="a" aria-label="Save">save</button>')
    old_state = await extract_state(session.page)
    assert old_state.selector_map[1].xpath == '//*[@id="a"]'

    await session.page.set_content(
        '<button id="other">new</button><button id="b" aria-label="Save">save</button>'
    )

    action = Action(name="click", params=ClickParams(index=1))
    recovered = await recover_stale(action, old_state, session.page)
    assert recovered is not None
    new_action, new_state = recovered
    assert new_action.params.index == 2  # relocalized via ax_name

    # Re-executing against the fresh state succeeds.
    result = await execute(new_action, session.page, new_state)
    assert result.success is True


@pytest.mark.asyncio
async def test_recover_stale_returns_none_when_element_gone(session):
    await session.page.set_content('<button id="a" aria-label="Save">save</button>')
    old_state = await extract_state(session.page)
    # Element removed entirely — no signal can relocalize it.
    await session.page.set_content("<button id='other'>new</button>")
    assert await recover_stale(Action(name="click", params=ClickParams(index=1)), old_state, session.page) is None


@pytest.mark.asyncio
async def test_agent_relocalizes_stale_index_using_previous_state(session):
    # Three buttons; "Save" is the third. Clicking it collapses the page so that
    # "Save" becomes the only (first) element. The model then emits the stale
    # index 3 from its *previous* perception; recovery must relocalize it using
    # the previous step's element (by accessible name) rather than giving up.
    await session.page.set_content(
        '<button id="x">x</button>'
        '<button id="y">y</button>'
        '<button id="save" aria-label="Save" onclick="collapse()">save</button>'
        '<script>'
        'function collapse(){'
        '  document.getElementById("x").remove();'
        '  document.getElementById("y").remove();'
        '  document.getElementById("save").id = "save2";'
        '}'
        '</script>'
    )

    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 3}},  # step 1: click Save -> collapses page
            {"name": "click", "params": {"index": 3}},  # step 2: stale index 3 -> relocalize
            {"name": "done", "params": {"success": True}},
        ]
    )
    agent = Agent(session=session, model=model, max_steps=10, max_failures=5)
    result = await agent.run(task="click save")

    assert result.done is True
    assert result.recovery_attempts == 1
    assert result.recoveries == 1  # recovery succeeded via the previous state's element
    assert result.history[1].results[0].success is True  # re-executed click succeeded


@pytest.mark.asyncio
async def test_recover_stale_falls_back_to_previous_state(session):
    # The model references index 3, which exists in the previous perception but
    # no longer in the current one (the page collapsed to a single element).
    await session.page.set_content(
        '<button id="x">x</button>'
        '<button id="y">y</button>'
        '<button id="save" aria-label="Save">save</button>'
    )
    previous_state = await extract_state(session.page)
    assert previous_state.selector_map[3].ax_name == "Save"

    await session.page.set_content('<button id="save2" aria-label="Save">save</button>')
    current_state = await extract_state(session.page)
    assert 3 not in current_state.selector_map

    action = Action(name="click", params=ClickParams(index=3))
    recovered = await recover_stale(action, current_state, session.page, previous_state=previous_state)
    assert recovered is not None
    new_action, _ = recovered
    assert new_action.params.index == 1  # relocalized via the previous state's element


@pytest.mark.asyncio
async def test_agent_attempts_stale_recovery_on_stale_failure(session, monkeypatch):
    calls: list[str] = []

    async def fake_recover(action, old_state, page, previous_state=None):
        calls.append(action.name)
        return None  # simulate: could not relocalize

    monkeypatch.setattr("minicua.controller.agent.recover_stale", fake_recover)

    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 999}},  # stale, not in selector map
            {"name": "done", "params": {"success": True}},
        ]
    )
    agent = Agent(session=session, model=model, max_steps=10, max_failures=5)
    result = await agent.run(task="x")
    assert result.done is True
    assert calls == ["click"]  # recovery was wired in for the stale failure


@pytest.mark.asyncio
async def test_agent_reobserves_and_replans_after_hallucinated_index(session):
    # A single "Submit" button (index 1). The model first hallucinates index 999,
    # which has no old element to relocalize (it never existed in the DOM).
    # Recovery must therefore escalate to re-observe + re-plan: the model is fed
    # the fresh DOM and emits the correct index 1, which then succeeds.
    await session.page.set_content('<button id="submit" aria-label="Submit">go</button>')
    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 999}},  # hallucinated index
            {"name": "click", "params": {"index": 1}},    # re-planned on fresh DOM
            {"name": "done", "params": {"success": True}},
        ]
    )
    agent = Agent(session=session, model=model, max_steps=10, max_failures=5)
    result = await agent.run(task="click submit")

    assert result.done is True
    assert result.recovery_attempts == 1
    assert result.recoveries == 1  # one successful re-observe + re-plan
    # The hallucinated click failed, then the re-planned click succeeded.
    assert result.history[0].results[0].success is False
    assert result.history[0].results[1].success is True


# --------------------------------------------------------------------------- #
# page-change guard (multi-action abort)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_aborts_multi_action_queue_on_page_change(session):
    model = FakeModel(
        responses=[
            [
                {"name": "navigate", "params": {"url": "data:text/html,<button id=b>ok</button>"}},
                {"name": "click", "params": {"index": 1}},  # grounded on the pre-navigation page
            ],
        ]
    )
    agent = Agent(session=session, model=model, max_steps=10)
    step = await agent.step()
    assert step.is_done is False
    assert step.page_changed is True
    assert len(step.results) == 1  # only the navigate executed; the click was aborted


# --------------------------------------------------------------------------- #
# loop detection (soft nudge)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_injects_loop_nudge(session):
    await session.page.set_content("<button id=b>x</button>")
    model = FakeModel(
        responses=[
            {"name": "click", "params": {"index": 1}},
            {"name": "click", "params": {"index": 1}},
            {"name": "click", "params": {"index": 1}},
            {"name": "done", "params": {"success": True}},
        ]
    )
    agent = Agent(session=session, model=model, max_steps=10, loop_threshold=2)
    result = await agent.run(task="x")
    assert result.done is True

    nudges = [
        m.content
        for call in model.calls
        for m in call[0]
        if m.role == "user" and "repeated" in m.content
    ]
    assert nudges


@pytest.mark.asyncio
async def test_agent_injects_replan_nudge_on_consecutive_failures(session):
    # Two non-stale failures (typing into a non-editable button) should push a
    # "REPLAN SUGGESTED" nudge so the model changes strategy instead of retrying.
    await session.page.set_content("<button id=b>click me</button>")
    model = FakeModel(
        responses=[
            {"name": "type", "params": {"index": 1, "text": "hi"}},  # not editable
            {"name": "type", "params": {"index": 1, "text": "hi"}},  # fails again
            {"name": "done", "params": {"success": True}},
        ]
    )
    agent = Agent(session=session, model=model, max_steps=10)
    result = await agent.run(task="x")
    assert result.done is True

    nudges = [
        m.content
        for call in model.calls
        for m in call[0]
        if m.role == "user" and "REPLAN SUGGESTED" in m.content
    ]
    assert nudges


@pytest.mark.asyncio
async def test_minimal_mode_disables_replan_nudge(session):
    await session.page.set_content("<button id=b>click me</button>")
    model = FakeModel(
        responses=[
            {"name": "type", "params": {"index": 1, "text": "hi"}},
            {"name": "type", "params": {"index": 1, "text": "hi"}},
            {"name": "done", "params": {"success": True}},
        ]
    )
    agent = Agent(session=session, model=model, recovery=False, max_steps=10, max_failures=5)
    result = await agent.run(task="x")

    nudges = [
        m.content
        for call in model.calls
        for m in call[0]
        if m.role == "user" and "REPLAN SUGGESTED" in m.content
    ]
    assert not nudges  # replan-on-stall is gated by the recovery master switch


# --------------------------------------------------------------------------- #
# crash recovery
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_recovers_from_crash(session, tmp_path):
    await session.page.set_content("<button id=b>x</button>")
    model = FakeModel(responses=[{"name": "done", "params": {"success": True}}])
    agent = Agent(session=session, model=model, checkpoint_dir=tmp_path)

    # Simulate the watchdog flagging a browser crash before the first step.
    agent._watchdog.crashed = True

    result = await agent.run(task="book a flight")
    assert result.done is True
    assert session.page is not None
    assert agent._watchdog.crashed is False  # recovered and reset
    assert agent.task == "book a flight"  # task state restored from checkpoint
